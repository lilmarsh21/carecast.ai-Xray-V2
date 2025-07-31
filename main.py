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

# ----------------------------
# 🔥 Heatmap overlay function
# ----------------------------
def apply_heatmap_overlay(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode('.png', overlay)
    encoded_overlay = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_overlay}"

# ----------------------------
# 📂 Read DICOM helper
# ----------------------------
def read_dicom_file(file_bytes):
    ds = pydicom.dcmread(BytesIO(file_bytes))
    pixel_array = ds.pixel_array

    # Normalize pixel data to 0-255
    pixel_array = pixel_array.astype(np.float32)
    pixel_array -= np.min(pixel_array)
    if np.max(pixel_array) != 0:
        pixel_array /= np.max(pixel_array)
    pixel_array *= 255.0

    image = pixel_array.astype(np.uint8)
    if len(image.shape) == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

# ----------------------------
# 🔑 Load config
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SECRET_KEY = os.getenv("UPLOAD_SECRET_KEY")

# ----------------------------
# 🚀 FastAPI app setup
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_reports = {}

# ----------------------------
# ✅ Upload endpoint
# ----------------------------
@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
    title: str = Form(...),
    user_meta: str = Form(...)
):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    image_data = await file.read()
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "No image received."})

    # Detect file type and convert
    filename = file.filename.lower()
    if filename.endswith(".dcm"):
        image = read_dicom_file(image_data)
        _, buffer = cv2.imencode('.png', image)
        mime_type = "image/png"
    else:
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        buffer = np_arr
        mime_type = file.content_type or "image/jpeg"

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    # Encode to base64
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    # Metadata injection
metadata = user_meta

    # System prompt
    system_prompt = (
    f"You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, and other medical imaging. "
    "You will be given both a detailed image and a patient metadata description. The metadata contains important clinical history, symptoms, or context that must inform your interpretation. "
    f"\n\nPATIENT METADATA:\n{user_meta}\n\n"
    "Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
    "You must always respond with a fully structured diagnostic report, even in cases where the image appears normal, incomplete, or of low quality. "
    "Do not provide disclaimers such as 'I’m unable to analyze this image.' Instead, deliver your best possible assessment based on available data. "
    "Structure your report using the following required sections:\n"
    "- **Findings** – A clear and itemized summary of all observed image features, including measurements, densities, anomalies, and any regions of interest.\n"
    "- **Impression** – A concise diagnostic interpretation or suspected condition based on the findings.\n"
    "- **Explanation** – A deeper clinical rationale for the impression, referencing anatomical or pathological details when appropriate.\n"
    "- **Recommended Care Plan** – Next steps for clinical follow-up, such as additional imaging, referrals, or urgent care if warranted.\n\n"
    "If image quality is limited or obscured, still provide a cautious but informative assessment based on visible regions.\n\n"
    "Always end your response with the following disclaimer: This report is created by CareCast.AI. Please consult a licensed medical professional for final diagnosis and treatment."
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}}
        ]}
    ],
    temperature=0.6,
    max_tokens=3000
)

        result = response.choices[0].message.content
        session_id = str(uuid.uuid4())
        last_reports[session_id] = result

        overlayed_image = apply_heatmap_overlay(image)

        return {
            "result": result,
            "session_id": session_id,
            "overlayed_image": overlayed_image
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
