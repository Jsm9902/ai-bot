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
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    def setup_engine(self):
        if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
            return None
        
        all_docs = []
        for filename in os.listdir(UPLOAD_DIR):
            if filename.endswith(".pdf"):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                # [복구] 원본 대소문자 유지 (임베딩 모델이 대소문자 맥락을 이해함)
                                cleaned_content = clean_text(text)
                                all_docs.append(Document(
                                    page_content=cleaned_content,
                                    metadata={"source": filename, "page": i}
                                ))
                    print(f"✅ {filename} 로드 완료")
                except Exception as e:
                    print(f"❌ {filename} 로드 실패: {str(e)}")
                    continue

        if not all_docs:
            return None
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_docs = splitter.split_documents(all_docs)
        
        vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=self.embeddings, 
            client_settings=Settings(is_persistent=True, anonymized_telemetry=False)
        )
        
        # [복구] K=15 설정 (키워드와 의미 검색의 조화)
        bm25 = BM25Retriever.from_documents(split_docs)
        bm25.k = 15 
        
        vector_r = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={
               "k": 15,           # 13에서 10으로 하향하여 정예화
                "fetch_k": 40,     # 너무 넓게 찾기보다 핵심 위주로 후보 추출
                "lambda_mult": 0.7 # 중복을 피하고 연관성을 유지하는 밸런스
            }
        )

        self.compression_retriever = EnsembleRetriever(
            retrievers=[bm25, vector_r], 
            weights=[0.55, 0.45] 
        )
        return self.compression_retriever

rag_engine = RAGEngine()