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
import requests
from fpdf import FPDF

# ✅ Load environment variables
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

sessions = {}

def apply_heatmap_overlay(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    _, buffer_overlay = cv2.imencode('.png', overlay)
    _, buffer_original = cv2.imencode('.png', image)
    return (
        f"data:image/png;base64,{base64.b64encode(buffer_overlay).decode()}",
        f"data:image/png;base64,{base64.b64encode(buffer_original).decode()}"
    )

def upload_to_wordpress_library(file_bytes, filename):
    wp_url = "https://yourdomain.com/wp-admin/admin-ajax.php?action=carecast_upload_direct_file"
    files = {'file': (filename, file_bytes, 'image/png')}
    try:
        response = requests.post(wp_url, files=files)
        return response.json()
    except:
        return {"error": "Failed to upload to WordPress"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), x_api_key: str = Header(...), title: str = Form("")):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    contents = await file.read()

    overlay_img, original_img = apply_heatmap_overlay(contents)
    if not overlay_img or not original_img:
        return JSONResponse(status_code=400, content={"error": "Image processing failed"})

    wp_result = upload_to_wordpress_library(contents, f"{uuid.uuid4()}.png")

    # ✅ Real GPT system prompt
    system_prompt = (
        "You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, and other medical imaging. "
        "Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
        "You must always respond with a fully structured diagnostic report, even in cases where the image appears normal, incomplete, or of low quality. "
        "Do not provide disclaimers such as 'I’m unable to analyze this image.' Instead, deliver your best possible assessment based on available data. "
        "Structure your report using the following required sections: "
        "- **Findings** – A clear and itemized summary of all observed image features, including measurements, densities, anomalies, and any regions of interest. "
        "- **Impression** – A concise diagnostic interpretation or suspected condition based on the findings. "
        "- **Explanation** – A deeper clinical rationale for the impression, referencing anatomical or pathological details when appropriate. "
        "- **Recommended Care Plan** – Next steps for clinical follow-up, such as additional imaging, referrals, or urgent care if warranted. "
        "If image quality is limited or obscured, still provide a cautious but informative assessment based on visible regions. "
        "Always end your response with the following disclaimer: This report is created by CareCast.AI. Please consult a licensed medical professional for final diagnosis and treatment."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please analyze this X-ray image and generate a full diagnostic report."}
            ],
            max_tokens=1200
        )
        final_report = response.choices[0].message.content.strip()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"report": final_report}

    return {
        "result": final_report,
        "session_id": session_id,
        "overlayed_image": overlay_img,
        "original_image": original_img,
        "media_library_url": wp_result.get("url")
    }

@app.post("/follow-up/")
async def follow_up(question: str = Form(...), session_id: str = Form(...), x_api_key: str = Header(...)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    base_report = sessions.get(session_id, {}).get("report", "")
    if not base_report:
        return {"follow_up_response": "❌ Session not found."}

    prompt = (
        f"You are a medical expert. Based on this previous report:\n\n{base_report}\n\n"
        f"The user has asked a follow-up question:\n\n{question}\n\n"
        "Please provide a clinical, helpful response."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        reply = response.choices[0].message.content.strip()
        return {"follow_up_response": reply}
    except Exception as e:
        return {"follow_up_response": f"❌ Error: {str(e)}"}

@app.post("/generate-pdf/")
async def generate_pdf(report_text: str = Form(...)):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in report_text.split("\\n"):
        pdf.multi_cell(0, 10, line)
    filename = f"/tmp/report_{uuid.uuid4().hex}.pdf"
    pdf.output(filename)

    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    os.remove(filename)
    return {"download_url": f"data:application/pdf;base64,{encoded}"}
"""

final_main_lines = final_main_with_prompt.strip().split("\n")
len(final_main_lines)
