from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from fpdf import FPDF
import uuid

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
        "You are a clinical radiologist, you double check everything for the best findings. You are being shown an X-ray, Ultrasound, MRI, or medical image, and your job is to provide a full analysis. "
        "You must always respond with a structured diagnostic report — Never say anything along the lines of I'm unable to analyze specific images directly, even if the findings are normal or limited.\n\n"
        "Format your response using these exact sections:\n"
        "- Findings\n- Impression\n- Explanation\n- Recommended Care Plan\n\n"
        "Always give a response. Do not say you cannot analyze. If image quality is poor, still attempt a limited assessment.\n"
        "End with: 'This report is created by CareCast.AI. Please consult a licensed medical professional for final diagnosis and treatment.'"
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

        return { "result": result, "session_id": session_id }

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
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        return { "follow_up_response": answer }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Generate PDF
@app.post("/generate-pdf/")
async def generate_pdf(report_text: str = Form(...)):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in report_text.split('\n'):
            pdf.multi_cell(0, 10, line)

        filename = f"{uuid.uuid4()}.pdf"
        filepath = f"/tmp/{filename}"
        pdf.output(filepath)

        return {
            "download_url": f"https://carecast-ai-xray-v2.onrender.com/download-pdf/{filename}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ Download PDF
@app.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    file_path = f"/tmp/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename="CareCast_Report.pdf", media_type='application/pdf')
