import os
import json
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from engine import rag_engine
from config import MODEL_NAME

async def stream_answer(query: str, history):
    if rag_engine.compression_retriever is None:
        rag_engine.setup_engine()

    # 기존 모델 사용
    llm = ChatOllama(model=MODEL_NAME, temperature=0.02, streaming=True)

    # 1. 문서 검색
    retrieved_docs = rag_engine.compression_retriever.invoke(query)
    
    # 2. 선 검증 (가중치 0.5:0.5 적용하여 정밀도 향상)
    filtered_docs = []
    temp_sources = []
    THRESHOLD = 0.05 # 백지선 학생 입학일자 등 짧은 팩트 통과를 위해 대폭 하향
    
    query_words = set([w for w in query.split() if len(w) >= 2])

    for i, doc in enumerate(retrieved_docs, 1):
        content_words = [w for w in doc.page_content.split() if len(w) >= 2]
        match_count = sum(1 for w in content_words if w in query_words)
        match_density = match_count / len(content_words) if content_words else 0
        
        # 엔진 순위 점수와 일치도 밀도를 반반씩 섞음
        search_rank_score = (11 - i) / 10
        final_score = (search_rank_score * 0.5) + (match_density * 0.5)

        if final_score >= THRESHOLD:
            filtered_docs.append(doc)
            temp_sources.append({
                "file": os.path.basename(doc.metadata.get('source', '')),
                "page": doc.metadata.get('page', 0) + 1,
                "score": final_score
            })

    # 3. 답변 생성
    system_prompt = "제공된 context만을 사용하여 질문에 답하세요.\n\n{context}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = create_stuff_documents_chain(llm, prompt)

    async for chunk in chain.astream({"input": query, "context": filtered_docs}):
        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"

    yield f"data: {json.dumps({'type': 'sources', 'sources': temp_sources})}\n\n"