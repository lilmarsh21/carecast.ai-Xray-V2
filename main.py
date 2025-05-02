from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    allow_headers=["*"],  # includes X-API-KEY
)

# ✅ Upload + Analyze
@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    print("SECRET_KEY:", SECRET_KEY)

    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    image_data = await file.read()
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "No image received."})

    mime_type = file.content_type or "image/jpeg"
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical AI assistant trained in radiology. "
                        "Analyze all visible bones, tissues, and abnormalities in the uploaded X-ray or MRI image. "
                        "Identify any fractures or broken bones, name the affected bones, describe dislocations, tumors, swelling, or anomalies. "
                        "After your findings, write a detailed radiologist-style report including:\n"
                        "- A clear explanation of what the abnormality is and why it may have happened\n"
                        "- Possible causes (trauma, disease, congenital, etc.)\n"
                        "- A proposed care plan: initial steps, referrals (e.g. orthopedic, neurologist), imaging follow-up, or treatment options\n"
                        "- Use structured headers: 'Findings', 'Impression', 'Explanation', 'Recommended Care Plan'\n"
                        "Always end with: 'This is an AI-generated report. Please consult a licensed medical professional for diagnosis and treatment.'"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and provide a full report."
                        },
                        {
                            "type": "image_url",
                            "image_url": { "url": image_url }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )

        result = response.choices[0].message.content
        return { "result": result }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ PDF Generator Endpoint
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
            "download_url": f"/download-pdf/{filename}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ PDF Download Endpoint
from fastapi.responses import FileResponse

@app.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    file_path = f"/tmp/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename="CareCast_Report.pdf", media_type='application/pdf')
