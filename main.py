@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), x_api_key: str = Header(...)):
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    image_data = await file.read()
    if not image_data:
        return JSONResponse(status_code=400, content={"error": "No image received."})

    # 🔹 Save original image as Base64
    mime_type = file.content_type or "image/jpeg"
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    original_image_url = f"data:{mime_type};base64,{image_base64}"

    # 🔹 Create heatmap overlay (Base64)
    overlay_img, overlayed_image = apply_heatmap_overlay(image_data)
    if not overlayed_image:
        return JSONResponse(status_code=500, content={"error": "Heatmap processing failed."})

    # 🔹 System prompt for GPT
    system_prompt = (
        "You are a highly experienced clinical radiologist specializing in the interpretation of X-rays, ultrasounds, MRIs, and other medical imaging. "
        "Your responsibility is to perform a comprehensive, high-detail analysis of the image provided, identifying all relevant abnormalities, patterns, and clinical indicators — including subtle or borderline findings. "
        "You must always respond with a fully structured diagnostic report, even in cases where the image appears normal, incomplete, or of low quality. "
        "Structure your report using the following required sections: "
        "- **Findings**\n- **Impression**\n- **Explanation**\n- **Recommended Care Plan**"
    )

    # 🔹 Send image to GPT for interpretation
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyze this X-ray and follow this structure:\n- **Findings**\n- **Impression**\n- **Explanation**\n- **Recommended Care Plan**"},
                {"type": "image_url", "image_url": {"url": original_image_url}}
            ]}
        ],
        max_tokens=2200
    )

    result = completion.choices[0].message.content.strip()
    session_id = str(uuid.uuid4())
    last_reports[session_id] = result
    last_images[session_id] = overlay_img

    # 🔹 Return both images
    return {
        "result": result,
        "session_id": session_id,
        "original_image": original_image_url,
        "overlayed_image": overlayed_image
    }
