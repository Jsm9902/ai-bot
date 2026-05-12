from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from config import UPLOAD_DIR
from routes import router

def create_app():
    app = FastAPI(title="AI PDF Analyzer")
    
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/pdf_files", StaticFiles(directory=UPLOAD_DIR), name="pdf_files")
    app.include_router(router)
    
    return app

app = create_app()