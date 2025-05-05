from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import os, shutil
from app.model_loader import load_model
from app.config import TRAIN_DIR

app = FastAPI(title="LLaVA Document Classifier API")
app.mount("/static", StaticFiles(directory="static"), name="static")

model, processor, device = load_model()

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        prompt = "Classify the type of document (e.g., invoice, receipt, ID card, contract, form, etc.)."
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=50)
        label = processor.decode(output[0], skip_special_tokens=True)
        return {"filename": file.filename, "classification": label}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/train")
async def upload_training_sample(file: UploadFile = File(...), label: str = Form(...)):
    label_dir = os.path.join(TRAIN_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    with open(os.path.join(label_dir, file.filename), "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"Saved under label '{label}'"}

@app.post("/retrain")
async def retrain_model():
    return {"message": "Retraining (stubbed). Integrate fine-tuning here."}