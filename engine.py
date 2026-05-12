import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from config import MODEL_NAME, DB_PATH, UPLOAD_DIR

class RAGEngine:
    def __init__(self):
        self.vectorstore = None
        self.compression_retriever = None

    def setup_engine(self):
        # 1. 문서 로딩 시작 (로그 출력)
        all_docs = []
        if os.path.exists(UPLOAD_DIR):
            for file in os.listdir(UPLOAD_DIR):
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(UPLOAD_DIR, file))
                    all_docs.extend(loader.load())
                    print(f"✅ {file} 로드 완료")

        # 2. 텍스트 분할 및 임베딩 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(all_docs)
        
        # [핵심] 기존 Bllossom 모델(MODEL_NAME)만 사용하도록 고정
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=DB_PATH
        )

        # 3. 검색 리트리버 구성
        compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        print("[시스템] 초기 문서 로드 완료! 챗봇이 준비되었습니다.\n")

rag_engine = RAGEngine()