from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
import os, base64

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
UPLOAD_SECRET = os.getenv("UPLOAD_SECRET")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        # ✅ Validate request secret
        if request.headers.get("X-API-KEY") != UPLOAD_SECRET:
            return JSONResponse(status_code=403, content={"error": "Unauthorized access"})

        image_data = await file.read()
        if not image_data:
            return JSONResponse(status_code=400, content={"error": "No image received."})

        mime_type = file.content_type or "image/jpeg"
        base64_img = base64.b64encode(image_data).decode("utf-8")
        image_url = f"data:{mime_type};base64,{base64_img}"

        # 🧠 Radiologist-style prompt
        response = client.chat.completions.create(
            model="gpt-4-turbo",
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
                        { "type": "text", "text": "Please analyze this image for broken bones, visible abnormalities, and provide a full medical interpretation and care plan." },
                        { "type": "image_url", "image_url": { "url": image_url } }
                    ]
                }
            ],
            max_tokens=2000
        )

        result = response.choices[0].message.content
        return { "result": result }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
