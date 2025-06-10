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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SECRET_KEY = os.getenv("UPLOAD_SECRET_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    image_url = f"data:{mime_type};base64,{image_base64}"

    overlay_img, overlayed_image = apply_heatmap_overlay(image_data)
    if not overlayed_image:
        return JSONResponse(status_code=500, content={"error": "Heatmap processing failed."})

    system_prompt = (
    "You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, and other medical imaging. "
    "Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
    "You must always respond with a fully structured diagnostic report, even in cases where the image appears normal, incomplete, or of low quality. "
    "Structure your report using the following required sections:\n"
    "- **Findings** – A clear and itemized summary of all observed issues.\n"
    "- **Impression** – A clinical interpretation summarizing the findings.\n"
    "- **Explanation** – Describe the reason for the impression and how it relates to the image.\n"
    "- **Recommended Care Plan** – Suggest next steps or referrals.\n\n"
    "Respond only in this format. Do not say you are unable to analyze the image. Do not refer the user to a radiologist. Output the full report every time."
)


    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze this X-ray and follow this structure:\n- **Findings**\n- **Impression**\n- **Explanation**\n- **Recommended Care Plan**"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ],
        max_tokens=1500
    )

    result = completion.choices[0].message.content.strip()
    session_id = str(uuid.uuid4())
    last_reports[session_id] = result
    last_images[session_id] = overlay_img

    return {
        "result": result,
        "session_id": session_id,
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
        max_tokens=2000
    )

    reply = response.choices[0].message.content.strip()
    return {"follow_up_response": reply}
