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
import pydicom
from io import BytesIO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SECRET_KEY = os.getenv("UPLOAD_SECRET_KEY")
session_data = {}

def apply_heatmap_overlay(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.png', overlay)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"

def read_dicom_file(file_bytes):
    ds = pydicom.dcmread(BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array -= np.min(pixel_array)
    if np.max(pixel_array) != 0:
        pixel_array /= np.max(pixel_array)
    pixel_array *= 255.0
    image = pixel_array.astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

@app.post("/start-session/")
async def start_session(x_api_key: str = Header(...), user_prompt: str = Form(...)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    session_id = str(uuid.uuid4())
    session_data[session_id] = {"user_meta": user_prompt}
    return {"session_id": session_id}

@app.post("/upload-xrays/")
async def upload_xrays(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    user_meta = session_data[session_id]["user_meta"]
    image_data = await file.read()
    filename = file.filename.lower()
    if filename.endswith(".dcm"):
        image = read_dicom_file(image_data)
        mime_type = "image/png"
    else:
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        mime_type = file.content_type or "image/jpeg"

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    system_prompt = (
        "You are a highly experienced clinical radiologist specializing in X-rays, ultrasounds, MRIs. "
        "Generate a full diagnostic report based on the patient description and medical image. "
        "Structure: **Findings**, **Impression**, **Explanation**, **Recommended Care Plan**. "
        "Always answer in detail. Never say 'insufficient data'."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_meta.strip()},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                ]
            }
        ],
        temperature=0.6,
        max_tokens=3000
    )

    result = response.choices[0].message.content
    overlayed_image = apply_heatmap_overlay(image)
    session_data[session_id]["report"] = result

    return {
        "result": result,
        "session_id": session_id,
        "overlayed_image": overlayed_image
    }
