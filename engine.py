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
                    # pdfplumber를 사용하여 텍스트 추출 (OCR 대안)
                    with pdfplumber.open(file_path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                # 정제 및 문서 객체화
                                cleaned_content = clean_text(text)
                                all_docs.append(Document(
                                    page_content=cleaned_content,
                                    metadata={"source": filename, "page": i}
                                ))
                    print(f"✅ {filename} 로드 성공")
                except Exception as e:
                    print(f"❌ {filename} 로드 실패: {str(e)}")
                    continue

        if not all_docs:
            return None
        
        # 텍스트 분할 (중첩도를 높여 맥락 유지)
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=300)
        split_docs = splitter.split_documents(all_docs)
        
        # 벡터 스토어 및 리트리버 설정
        vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=self.embeddings, 
            client_settings=Settings(is_persistent=True, anonymized_telemetry=False)
        )
        
        bm25 = BM25Retriever.from_documents(split_docs)
        vector_r = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        self.compression_retriever = EnsembleRetriever(
            retrievers=[bm25, vector_r], 
            weights=[0.4, 0.6]
        )
        return self.compression_retriever

rag_engine = RAGEngine()