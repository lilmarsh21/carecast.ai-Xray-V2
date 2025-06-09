from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import uuid
import cv2
import numpy as np

load_dotenv()

app = FastAPI()  # ✅ MUST come before any @app decorators

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SECRET_KEY = os.getenv("UPLOAD_SECRET_KEY")

last_reports = {}
last_images = {}

def apply_heatmap_overlay(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.png', overlay)
    encoded_overlay = base64.b64encode(buffer).decode('utf-8')
    return overlay, f"data:image/png;base64,{encoded_overlay}"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    image_data = await file.read()
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "No image received."})

    mime_type = file.content_type or "image/jpeg"
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    original_image_url = f"data:{mime_type};base64,{image_base64}"

    overlay_img, overlayed_image = apply_heatmap_overlay(image_data)
    if not overlayed_image:
        return JSONResponse(status_code=500, content={"error": "Heatmap processing failed."})

    system_prompt = (
        "You are a board-certified radiologist with over 20 years of experience. "
        "Your role is to interpret high-resolution X-rays, MRIs, and ultrasound images with the highest level of clinical accuracy. "
        "When analyzing the image, always assume the reader is a physician or specialist who expects a complete and professional-level report. "
        "Do not simplify the language. Use real medical terminology, describe anatomical regions in detail, and reference all visible structures, even if they appear normal. "
        "When abnormalities are found, include likely etiologies, comparative severity, and clinical implications. "
        "Structure the report in the following format:\n\n"
        "- **Findings**: Describe everything visible in the image, both normal and abnormal.\n"
        "- **Impression**: Provide a differential diagnosis or leading conclusions based on the image.\n"
        "- **Explanation**: Support your impression with reasoning and references to the image.\n"
        "- **Recommended Care Plan**: Suggest the next steps for diagnosis or treatment, and what kind of specialist should be consulted."
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze this X-ray and follow this structure:\n- **Findings**\n- **Impression**\n- **Explanation**\n- **Recommended Care Plan**"},
                {"type": "image_url", "image_url": {"url": original_image_url}}
            ]}
        ],
        max_tokens=4096,
        temperature=0.2
    )

    result = completion.choices[0].message.content.strip()
    session_id = str(uuid.uuid4())
    last_reports[session_id] = result
    last_images[session_id] = overlay_img

    return {
        "result": result,
        "session_id": session_id,
        "original_image": original_image_url,
        "overlayed_image": overlayed_image
    }

@app.post("/follow-up/")
async def follow_up(question: str = Form(...), session_id: str = Form(...), x_api_key: str = Header(...)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    previous_report = last_reports.get(session_id)
    if not previous_report:
        raise HTTPException(status_code=404, detail="Session not found")

    follow_up_prompt = (
        "You are continuing from a previous radiology report. Use the findings below as context and answer the user's follow-up question as a radiologist.\n\n"
        f"{previous_report}\n\nQuestion: {question}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a board-certified radiologist providing medical insight."},
            {"role": "user", "content": follow_up_prompt}
        ],
        max_tokens=1200,
        temperature=0.2
    )

    reply = response.choices[0].message.content.strip()
    return {"follow_up_response": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
