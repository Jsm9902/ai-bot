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
    
    # 2. LLM 설정
    llm = ChatOllama(model=MODEL_NAME, temperature=0.02, streaming=True)

    # 3. 문서 검색
    retrieved_docs = rag_engine.compression_retriever.invoke(query)
    
    # 상위 12개 전달 (안정적인 데이터 풀 확보)
    filtered_docs = retrieved_docs[:12] if retrieved_docs else []

    print(f"\n{'='*85}\n[STEP 1] 엔진 검색 후보군 (TOP {len(filtered_docs)})")
    for i, doc in enumerate(filtered_docs, 1):
        fname = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 0) + 1
        print(f"  {i:>2}순위: {fname:<40} (p.{page})")
    print(f"{'='*85}")

    # 4. 답변 생성 및 스트리밍
    chain = create_stuff_documents_chain(llm, get_qa_prompt())
    full_response = ""
    
    async for chunk in chain.astream({
        "input": query, 
        "chat_history": history.messages, 
        "context": filtered_docs
    }):
        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
        full_response += chunk

    # 5. STEP 2: 밀도 분석 및 하이브리드 가중치 정렬 (보정 로직)
    temp_sources = []
    seen = set()
    THRESHOLD = 0.11 # 요약 답변을 위해 임계값을 약간 낮추어 정답 누락 방지
    response_words = set([w for w in full_response.split() if len(w) >= 2])

    print(f"\n[STEP 2] 하이브리드 가중치 정렬 분석 (Threshold: {THRESHOLD})")
    print(f"{'-'*85}")
    
    for i, d in enumerate(filtered_docs, 1): # i는 검색 엔진이 매긴 순위 (1~12)
        p = d.metadata.get('page', 0) + 1
        f = os.path.basename(d.metadata.get('source', ''))
        content_words = [w for w in d.page_content.split() if len(w) >= 2]
        if not content_words: continue
        
        # 밀도(Density) 계산
        match_count = sum(1 for w in content_words if w in response_words)
        match_density = match_count / len(content_words)
        
        # [핵심 보정 로직]
        # 1. 검색 순위 점수 (1순위=1.0, 12순위=약 0.08)
        search_rank_score = (len(filtered_docs) - i + 1) / len(filtered_docs)
        
        # 2. 최종 점수 산출 (검색 순위 60% + 매칭 밀도 40%)
        # 이를 통해 '우연히 단어만 겹치는 노이즈'는 엔진 순위가 낮아 뒤로 밀립니다.
        final_score = (search_rank_score * 0.5) + (match_density * 0.5)

        status = "✅ PASS" if match_density >= THRESHOLD else "❌ DROP"
        print(f"{status:^8} | {f[:30]:<35} (p.{p}) | 밀도:{match_density:.4f} | 보정점수:{final_score:.4f}")

        if match_density >= THRESHOLD and f"{f}_{p}" not in seen:
            temp_sources.append({
                "file": f, 
                "page": p, 
                "score": final_score, # 정렬 기준
                "snippet": d.page_content[:150].replace('\n', ' ') + "..."
            })
            seen.add(f"{f}_{p}")
    
    # 보정된 최종 점수(score)를 기준으로 내림차순 정렬
    temp_sources.sort(key=lambda x: x['score'], reverse=True)

    # 최종 sources 리스트 생성
    sources = [
        {"file": s['file'], "page": s['page'], "snippet": s['snippet']} 
        for s in temp_sources
    ]
    
    print(f"{'-'*85}")
    print(f"최종 하이브리드 정렬 완료: {len(sources)}건 리스트업")
    print(f"{'-'*85}\n")

    # 대화 기록 저장
    history.add_user_message(query)
    history.add_ai_message(full_response)
    
    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
    yield "data: [DONE]\n\n"