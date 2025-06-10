from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import uuid
import cv2
import numpy as np
import requests

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SECRET_KEY = os.getenv("UPLOAD_SECRET_KEY")
WP_API_TOKEN = os.getenv("WP_API_TOKEN")  # 🔒 WordPress JWT Token

app = FastAPI()

# Serve uploaded files
UPLOAD_DIR = "uploaded_xrays"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploaded_xrays", StaticFiles(directory=UPLOAD_DIR), name="uploaded_xrays")

# Enable CORS
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

def upload_to_wp(file_path, filename):
    with open(file_path, 'rb') as f:
        headers = {
            'Authorization': f'Bearer {WP_API_TOKEN}',
            'Content-Disposition': f'attachment; filename={filename}'
        }
        response = requests.post(
            'https://doc.carecast.ai/wp-json/wp/v2/media',
            headers=headers,
            files={'file': (filename, f)}
        )
        if response.status_code == 201:
            return response.json().get("source_url")
        return None

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

    image_data = await file.read()
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "No image received."})

    # Save file locally
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(image_data)

    # Upload to WordPress Media Library
    public_url = upload_to_wp(file_path, filename)
    if not public_url:
        return JSONResponse(status_code=500, content={"error": "Failed to upload to WordPress."})

    # Generate overlay
    overlay_img, overlayed_image = apply_heatmap_overlay(image_data)
    if not overlayed_image:
        return JSONResponse(status_code=500, content={"error": "Heatmap processing failed."})

    # Prompt for GPT
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

    # GPT-4o call
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze this X-ray and follow the structure provided."},
                {"type": "image_url", "image_url": {"url": public_url}}
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
        "overlayed_image": overlayed_image,
        "image_url": public_url
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
