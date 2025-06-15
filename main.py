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

def apply_heatmap_overlay(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.png', overlay)
    encoded_overlay = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_overlay}"


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
    image_url = f"data:{mime_type};base64,{image_base64}"

    system_prompt = (
    "You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, CT scans, mammograms, and other medical imaging. "
    "Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
    "\n\nStructure your response using ALL of the following sections. "
    "**Include the section title followed by the definition, then your content. Example:**\n"
    "**Findings – A clear and itemized summary...**\n[your findings here]\n"
    "**Impression – A concise diagnostic interpretation...**\n[your impression here]\n"
    "**Explanation – A deeper clinical rationale...**\n[your explanation here]\n"
    "**Recommended Care Plan – Next steps...**\n[your care plan here]\n\n"
    "If image quality is limited or obscured, still provide a cautious but informative assessment based on visible regions. "
    "Use professional clinical reasoning and describe any features you *can* identify, even if limited.\n"
    
    "Always end your response with the following disclaimer: This report is created by CareCast.AI. Please consult a licensed medical professional for final diagnosis and treatment."
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

        overlayed_image = apply_heatmap_overlay(image_data)
        return { "result": result, "session_id": session_id, "overlayed_image": overlayed_image }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Follow-up endpoint
@app.post("/follow-up/")
async def follow_up(
    question: str = Form(...),
    session_id: str = Form(...),
    x_api_key: str = Header(...)
):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    if session_id not in last_reports:
        return JSONResponse(status_code=400, content={"error": "Invalid session ID"})

    try:
        system_prompt = (
            "You are a clinical radiologist AI following up on a previously analyzed X-ray or MRI. "
            "You will be asked additional questions and must reply in a professional, medically accurate tone."
        )

        messages = [
            { "role": "system", "content": system_prompt },
            { "role": "assistant", "content": last_reports[session_id] },
            { "role": "user", "content": question }
        ]

        response = client.chat.completions.create(
            model="GPT-4-1106-preview",
            messages=messages,
            max_tokens=1500
        )

        answer = response.choices[0].message.content
        return { "follow_up_response": answer }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
