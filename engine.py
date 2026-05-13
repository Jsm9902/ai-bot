import os
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from chromadb.config import Settings
from config import DB_PATH, UPLOAD_DIR
from utils import clean_text

class RAGEngine:
    def __init__(self):
        self.compression_retriever = None
        # [수정] 사용자님 환경에 맞는 전용 임베딩 모델로 고정
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    def setup_engine(self):
        if self.compression_retriever is not None:
            return self.compression_retriever

        if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
            return None
        
        print(f"\n[시스템] '{self.embeddings.model}' 모델 기반 인덱싱을 시작합니다...")
        
        all_docs = []
        for filename in os.listdir(UPLOAD_DIR):
            if filename.endswith(".pdf"):
                try:
                    with pdfplumber.open(os.path.join(UPLOAD_DIR, filename)) as pdf:
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                all_docs.append(Document(
                                    page_content=clean_text(text),
                                    metadata={"source": filename, "page": i}
                                ))
                    print(f"✅ {filename} 로드 완료")
                except Exception as e:
                    print(f"❌ {filename} 실패: {e}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=300)
        split_docs = splitter.split_documents(all_docs)
        
        vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=self.embeddings,
            client_settings=Settings(is_persistent=True, anonymized_telemetry=False)
        )
        
        # [수정] 사용자님의 강력한 앙상블 검색 설정 복구
        bm25 = BM25Retriever.from_documents(split_docs)
        bm25.k = 15 
        
        vector_r = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 15, "fetch_k": 40, "lambda_mult": 0.7}
        )

        self.compression_retriever = EnsembleRetriever(
            retrievers=[bm25, vector_r], 
            weights=[0.7, 0.3] 
        )
        print("[시스템] 준비 완료!\n")
        return self.compression_retriever

rag_engine = RAGEngine()