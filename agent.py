import json
import os
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import MODEL_NAME
from engine import rag_engine
from prompts import get_qa_prompt

async def stream_answer(query: str, history):
    # 1. 엔진 초기화 확인
    if rag_engine.compression_retriever is None:
        rag_engine.setup_engine()
    
    if not rag_engine.compression_retriever:
        yield f"data: {json.dumps({'type': 'content', 'delta': '업로드된 문서가 없습니다.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # LLM 설정 (온도 0으로 고정하여 일관성 유지)
    llm = ChatOllama(model=MODEL_NAME, temperature=0, streaming=True)

    # 2. 문서 검색 (TOP 12 후보군 추출)
    retrieved_docs = rag_engine.compression_retriever.invoke(query)
    filtered_docs = retrieved_docs[:12] 

    # [로그] STEP 1: 검색 결과 요약
    print(f"\n{'='*85}\n[STEP 1] 엔진 검색 후보군 (TOP 12)")
    for i, doc in enumerate(filtered_docs, 1):
        fname = os.path.basename(doc.metadata.get('source', 'unknown'))
        print(f"  {i:>2}순위: {fname:<30} (p.{doc.metadata.get('page', 0)+1})")
    print(f"{'='*85}")

    # 3. 답변 생성 및 스트리밍
    chain = create_stuff_documents_chain(llm, get_qa_prompt())
    full_response = ""
    
    async for chunk in chain.astream({
        "input": query, 
        "chat_history": history.messages, 
        "context": filtered_docs
    }):
        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
        full_response += chunk

    # 4. STEP 2: 답변-문서 매칭 밀도 분석 및 시각화
    sources = []
    seen = set()
    response_words = set([w for w in full_response.split() if len(w) >= 2])

    print(f"\n[STEP 2] 답변-문서 매칭 밀도 분석 (Threshold: 0.14)")
    print(f"{'-'*85}")
    print(f"{'상태':^8} | {'페이지 정보':^25} | {'밀도(Density)':^15} | {'단어 일치율':^15}")
    print(f"{'-'*85}")
    
    for d in filtered_docs:
        p = d.metadata.get('page', 0) + 1
        f = os.path.basename(d.metadata.get('source', ''))
        content_words = [w for w in d.page_content.split() if len(w) >= 2]
        
        if not content_words: continue
        
        match_count = sum(1 for w in content_words if w in response_words)
        match_density = match_count / len(content_words)
        
        # 시각화 상태 설정
        if match_density >= 0.14:
            status = "✅ PASS"
        else:
            status = "❌ DROP"
            
        short_fname = (f[:20] + '..') if len(f) > 20 else f
        page_info = f"{short_fname} (p.{p})"

        # 테이블 형식 출력
        print(f"{status:^8} | {page_info:<25} | {match_density:^15.4f} | {match_count:>5} / {len(content_words):<5}")

        # 출처 리스트 추가
        if match_density >= 0.14 and f"{f}_{p}" not in seen:
            sources.append({
                "file": f, 
                "page": p, 
                "snippet": d.page_content[:150].replace('\n', ' ') + "..."
            })
            seen.add(f"{f}_{p}")
    
    print(f"{'-'*85}\n")

    # 대화 기록 갱신 및 출처 전송
    history.add_user_message(query)
    history.add_ai_message(full_response)
    
    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
    yield "data: [DONE]\n\n"