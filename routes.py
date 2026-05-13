import os, shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_community.chat_message_histories import ChatMessageHistory

from config import UPLOAD_DIR, MAX_STORAGE_MB
from utils import get_total_size_mb
from engine import rag_engine
from agent import stream_answer

router = APIRouter()
templates = Jinja2Templates(directory="templates")
chat_histories = {}

def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/files")
async def list_files():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]
    return {"files": files, "current_size": round(get_total_size_mb(), 2), "max_size": MAX_STORAGE_MB}

@router.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    if get_total_size_mb() > MAX_STORAGE_MB:
        raise HTTPException(status_code=400, detail="Capacity Exceeded")
    for file in files:
        with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as b:
            shutil.copyfileobj(file.file, b)
    rag_engine.setup_engine()
    return {"status": "ok"}

@router.delete("/delete/{filename}")
async def delete_pdf(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        rag_engine.setup_engine()
        return {"status": "deleted"}
    raise HTTPException(status_code=404)

@router.post("/ask")
async def ask(question: str = Form(...), session_id: str = Form("default")):
    history = get_session_history(session_id)
    return StreamingResponse(stream_answer(question, history), media_type="text/event-stream")