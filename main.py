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
    "You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, and other medical imaging. Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
    "You must always respond with a fully structured diagnostic report, even in cases where the image appears normal, incomplete, or of low quality. Do not provide disclaimers such as 'I’m unable to analyze this image.' Instead, deliver your best possible assessment based on available data."
    "Structure your report using the following required sections: "
    "- Findings – A clear and itemized summary of all observed image features, including measurements, densities, anomalies, and any regions of interest. "
    "- Impression – A concise diagnostic interpretation or suspected condition based on the findings. "
    "- Explanation – A deeper clinical rationale for the impression, referencing anatomical or pathological details when appropriate. "
    "- Recommended Care Plan – Next steps for clinical follow-up, such as additional imaging, referrals, or urgent care if warranted. "
    "If image quality is limited or obscured, still provide a cautious but informative assessment based on visible regions. "
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
