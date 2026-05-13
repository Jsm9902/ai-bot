import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from config import UPLOAD_DIR
from routes import router
from engine import rag_engine  # 엔진 인스턴스 임포트

# 1. 서버 시작과 종료 시 실행될 수명 주기(Lifespan) 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [서버 시작 시점]
    print("\n" + "="*60)
    print("[시스템] 서버 가동을 감지했습니다. 초기 문서 인덱싱을 시작합니다.")
    try:
        # 가동 시 data 폴더에 있는 파일들을 읽어 엔진을 미리 세팅합니다.
        # 이 작업 덕분에 사용자가 접속했을 때 첫 질문 응답이 빨라집니다.
        rag_engine.setup_engine()
        print("[시스템] 초기 문서 로드 완료! 챗봇이 준비되었습니다.")
    except Exception as e:
        print(f"[시스템] 서버 시작 중 문서 로드 오류 발생: {e}")
    print("="*60 + "\n")
    
    yield
    
    # [서버 종료 시점]
    print("\n[시스템] 서버를 종료합니다.")
    pass

def create_app():
    # 2. FastAPI 인스턴스 생성 시 lifespan 설정을 연결합니다.
    app = FastAPI(
        title="AI PDF Analyzer",
        description="PDF 분석 및 RAG 기반 대화 시스템",
        lifespan=lifespan
    )
    
    # 정적 파일 및 업로드 파일 경로 마운트
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/pdf_files", StaticFiles(directory=UPLOAD_DIR), name="pdf_files")
    
    # API 라우터 등록 (routes.py 연결)
    app.include_router(router)
    
    return app

# 최종 앱 인스턴스 생성
app = create_app()