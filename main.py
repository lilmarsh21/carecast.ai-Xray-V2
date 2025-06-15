from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from fpdf import FPDF
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

# 🧠 Store reports per session (in-memory)
last_reports = {}

# ✅ Upload & Analyze
@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    image_data = await file.read()
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "No image received."})

    mime_type = file.content_type or "image/jpeg"
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"  # ✅ used for GPT
    preview_image = image_url  # ✅ returned to frontend for preview

    system_prompt = (
        "You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, and other medical imaging. Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
        "You must always respond with a fully structured diagnostic report, even in cases where the image appears normal, incomplete, or of low quality. Do not provide disclaimers such as 'I’m unable to analyze this image.' Instead, deliver your best possible assessment based on available data."
        "\n\nStructure your response using ALL of the following sections. "
        "**Include the section title followed by the definition, then your content. Example:**\n"
        "**Findings – A clear and itemized summary...**\n[your findings here]\n"
        "**Impression – A concise diagnostic interpretation...**\n[your impression here]\n"
        "**Explanation – A deeper clinical rationale...**\n[your explanation here]\n"
        "**Recommended Care Plan – Next steps...**\n[your care plan here]\n\n"
        "Do not skip any section, even if information is limited. Always return all 4 sections with headings and definitions."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": system_prompt },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Please analyze this image." },
                        { "type": "image_url", "image_url": { "url": image_url } }
                    ]
                }
            ],
            max_tokens=2000
        )

        result = response.choices[0].message.content
        session_id = str(uuid.uuid4())
        last_reports[session_id] = result

        return {
            "result": result,
            "session_id": session_id,
            "preview_image": preview_image  # ✅ show in frontend
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
