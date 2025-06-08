from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddlewareAdd commentMore actions
from fastapi.responses import JSONResponse, FileResponse
@@ -10,6 +9,7 @@
import cv2
import numpy as np
from io import BytesIO
from fpdf import FPDF

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
@@ -25,25 +25,33 @@
    allow_headers=["*"],
)

# In-memory session tracking
last_reports = {}

def apply_heatmap_overlay(image_bytes):
COLORMAP_OPTIONS = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "cool": cv2.COLORMAP_COOL,
    "bone": cv2.COLORMAP_BONE,
    "winter": cv2.COLORMAP_WINTER
}

def apply_heatmap_overlay(image_bytes, map_name="jet"):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cmap = COLORMAP_OPTIONS.get(map_name.lower(), cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(gray, cmap)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode('.png', overlay)
    encoded_overlay = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_overlay}"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...)):
async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...), map: str = "jet"):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

@@ -55,12 +63,10 @@ async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    # Apply heatmap overlay
    overlayed_image = apply_heatmap_overlay(image_data)
    overlayed_image = apply_heatmap_overlay(image_data, map)
    if not overlayed_image:
        return JSONResponse(status_code=500, content={"error": "Heatmap processing failed."})

    # Generate report from OpenAI
    system_prompt = (
        "You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, and other medical imaging. "
        "Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
@@ -73,8 +79,6 @@ async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...
        "- **Recommended Care Plan** – Suggest next steps or referrals."
    )

    user_prompt = f"This is an encoded image for diagnosis: {image_url}"

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
@@ -86,24 +90,54 @@ async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...

    result = completion.choices[0].message.content.strip()
    session_id = str(uuid.uuid4())
    last_reports[session_id] = result
    last_reports[session_id] = {"text": result, "image": overlayed_image}

    return {
        "result": result,
        "session_id": session_id,
        "overlayed_image": overlayed_image
    }

@app.post("/follow-up/")
async def follow_up(question: str = Form(...), session_id: str = Form(...), x_api_key: str = Header(...)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    previous = last_reports.get(session_id)
    if not previous:
        return {"follow_up_response": "❌ Session expired or not found."}

    followup = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a clinical radiologist continuing a follow-up diagnosis."},
            {"role": "user", "content": previous["text"]},
            {"role": "user", "content": question}
        ],
        max_tokens=600
    )

    return {"follow_up_response": followup.choices[0].message.content.strip()}

# ✅ PDF Generation
@app.post("/generate-pdf/")
async def generate_pdf(report_text: str = Form(...)):
async def generate_pdf(report_text: str = Form(...), image_url: str = Form(None)):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    if image_url and image_url.startswith("data:image"):
        header, b64_data = image_url.split(",", 1)
        binary = base64.b64decode(b64_data)
        temp_path = f"/tmp/image_{uuid.uuid4().hex}.png"
        with open(temp_path, "wb") as f:
            f.write(binary)
        pdf.image(temp_path, x=10, y=10, w=180)
        pdf.ln(85)

    for line in report_text.splitlines():
        pdf.multi_cell(0, 10, line)

    filename = f"report_{uuid.uuid4().hex}.pdf"
    filepath = f"/tmp/{filename}"
    pdf.output(filepath)
