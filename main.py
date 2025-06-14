# Rebuild main.py with:
# - Full structure (approx. 160 lines)
# - WordPress Media Library upload
# - Return both overlayed and original image
# - Matches user's expected formatting and inline documentation style

final_main_py = """
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

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Session memory
sessions = {}

# ✅ Apply heatmap + return original image
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
    overlay_b64 = base64.b64encode(buffer_overlay).decode()
    original_b64 = base64.b64encode(buffer_original).decode()
    return f"data:image/png;base64,{overlay_b64}", f"data:image/png;base64,{original_b64}"

# ✅ Upload image to WordPress library via AJAX endpoint
def upload_to_wordpress_library(file_bytes, filename):
    wp_url = "https://yourdomain.com/wp-admin/admin-ajax.php?action=carecast_upload_direct_file"
    files = {'file': (filename, file_bytes, 'image/png')}
    try:
        response = requests.post(wp_url, files=files)
        return response.json()
    except:
        return {"error": "Failed to upload to WordPress"}

# ✅ Upload Endpoint
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), x_api_key: str = Header(...), title: str = Form("")):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # 🔍 Read file contents
    contents = await file.read()

    # 🔥 Generate heatmap and original
    overlay_img, original_img = apply_heatmap_overlay(contents)
    if not overlay_img or not original_img:
        return JSONResponse(status_code=400, content={"error": "Image processing failed"})

    # 📤 Upload original to WordPress Media Library
    wp_result = upload_to_wordpress_library(contents, f"{uuid.uuid4()}.png")

    # 🧠 Dummy AI response
    final_report = (
        "**Findings**\n- Patchy opacity in lower left lobe.\n\n"
        "**Impression**\n- Possible mild pneumonia or fluid retention.\n\n"
        "**Explanation**\n- Radiographic signs suggest infection-related infiltration.\n\n"
        "**Recommended Care Plan**\n- Schedule chest CT, clinical correlation advised."
    )

    # 🔐 Generate session ID
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"report": final_report}

    return {
        "result": final_report,
        "session_id": session_id,
        "overlayed_image": overlay_img,
        "original_image": original_img,
        "media_library_url": wp_result.get("url")
    }

# ✅ Follow-up Question Endpoint
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

# ✅ PDF Generation Endpoint
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

final_main_py_lines = final_main_py.strip().split("\n")
len(final_main_py_lines)
