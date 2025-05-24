from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from app.pdf_Reader import extract_text_from_pdf
from app.rag_engine import load_pdf_to_vectorstore, ask_question
import os

app = FastAPI()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Optional: extract text for any pre-processing (if needed)
    # text = extract_text_from_pdf(file_path)

    # ✅ FIXED: Pass file_path instead of text
    load_pdf_to_vectorstore(file_path)
    os.remove(file_path)  # Clean up the temp file

    return {"message": "PDF processed successfully."}

@app.post("/ask/")
async def ask(query: str = Form(...)):
    answer = ask_question(query)
    return {"answer": answer}
