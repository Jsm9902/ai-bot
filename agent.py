import json
import os
import re
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import MODEL_NAME
from engine import rag_engine
from prompts import get_qa_prompt

async def stream_answer(query: str, history):
    if rag_engine.compression_retriever is None:
        rag_engine.setup_engine()
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0.02, streaming=True)

    # 1. 문서 검색 (STEP 1)
    retrieved_docs = rag_engine.compression_retriever.invoke(query)
    all_candidates = retrieved_docs[:12] if retrieved_docs else []

    # 키워드 추출 (특수문자 포함 대응)
    query_keywords = [w.strip() for w in re.split(r'\s+', query) if len(w) >= 2]

    print(f"\n{'='*85}\n[STEP 1] 엔진 검색 후보군 (TOP {len(all_candidates)})")
    for i, doc in enumerate(all_candidates, 1):
        fname = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 0) + 1
        print(f"  {i:>2}순위: {fname:<40} (p.{page})")
    print(f"{'='*85}")

    # 2. 사전 컨텍스트 전달 (AI가 모든 정보를 보도록 제한 완화)
    # 엔진이 찾은 12개 후보를 AI가 일단 모두 보게 하여 "정보 없음" 오류를 방지합니다.
    context_docs = all_candidates

    # 3. 답변 생성 스트리밍
    chain = create_stuff_documents_chain(llm, get_qa_prompt())
    full_response = ""
    async for chunk in chain.astream({
        "input": query, 
        "chat_history": history.messages, 
        "context": context_docs
    }):
        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
        full_response += chunk

    # 4. [STEP 2] 후행 정밀 검증 (요청하신 0.05 격차 및 기준점 보정)
    temp_sources = []
    seen = set()
    response_words = set([w for w in full_response.split() if len(w) >= 2])

    densities = []
    for d in all_candidates:
        content_words = [w for w in d.page_content.split() if len(w) >= 2]
        if not content_words:
            densities.append(0); continue
        m_count = sum(1 for w in content_words if w in response_words)
        densities.append(m_count / len(content_words))

    # [수정 사항] 전체 후보 중 '가장 높은 밀도'를 기준으로 설정
    absolute_max_density = max(densities) if densities else 0
    # [수정 사항] 요청하신 초정밀 격차 0.05 적용
    STRICT_GAP = 0.05 

    print(f"\n[STEP 2] 초정밀 격차 필터링 (최대 밀도: {absolute_max_density:.4f} / Gap: {STRICT_GAP})")
    print(f"{'-'*85}")
    
    for i, d in enumerate(all_candidates, 1):
        f = os.path.basename(d.metadata.get('source', ''))
        p = d.metadata.get('page', 0) + 1
        match_density = densities[i-1]
        
        search_rank_score = (len(all_candidates) - i + 1) / len(all_candidates)
        final_score = (search_rank_score * 0.5) + (match_density * 0.5)
        
        has_match = (match_density > 0)
        has_query_keyword = any(kw.lower() in d.page_content.lower() for kw in query_keywords)

        # --- [요청하신 정밀 통과 로직] ---
        # 1. 진짜 최대 밀도 문서는 PASS
        if match_density == absolute_max_density and has_match:
            is_relevant = True
        # 2. 나머지는 키워드가 있고 격차가 0.05 이내여야 함
        else:
            is_relevant = has_query_keyword and (absolute_max_density - match_density <= STRICT_GAP)
            # 엔진 1순위 보호 (최대 밀도의 30% 수준 유지 시)
            if i == 1 and has_match and (match_density >= absolute_max_density * 0.3):
                is_relevant = True

        status = "✅ PASS" if (final_score >= 0.35 and is_relevant) else "❌ DROP"
        print(f"{status:^8} | {f[:30]:<35} (p.{p}) | 밀도:{match_density:.4f} | 보정점수:{final_score:.4f}")

        if status == "✅ PASS" and f"{f}_{p}" not in seen:
            temp_sources.append({
                "file": f, "page": p, "score": final_score,
                "snippet": d.page_content[:150].replace('\n', ' ') + "..."
            })
            seen.add(f"{f}_{p}")

    temp_sources.sort(key=lambda x: x['score'], reverse=True)
    sources = [{"file": s['file'], "page": s['page'], "snippet": s['snippet']} for s in temp_sources]
    
    yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
    yield "data: [DONE]\n\n"