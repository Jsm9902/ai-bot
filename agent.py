import json
import os
import re
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from config import MODEL_NAME
from engine import rag_engine
from prompts import get_qa_prompt, QUERY_EXPANSION_PROMPT

def calculate_fuzzy_density(target_content, response_text):
    if not response_text or len(response_text.strip()) < 2:
        return 0.0
    clean_target = re.sub(r'\s+', '', target_content).lower()
    response_words = [w for w in response_text.split() if len(w) >= 2]
    if not response_words: return 0.0
    match_count = sum(1 for w in response_words if w.replace(" ", "").lower() in clean_target)
    return match_count / len(response_words)

async def stream_answer(query: str, history):
    if rag_engine.compression_retriever is None:
        rag_engine.setup_engine()
    
    # [1단계] 의미 기반 쿼리 확장 및 터미널 출력
    expansion_llm = ChatOllama(model=MODEL_NAME, temperature=0)
    expansion_prompt = PromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
    expansion_chain = expansion_prompt | expansion_llm
    expanded_result = await expansion_chain.ainvoke({"input": query})
    expanded_query = expanded_result.content.strip()
    
    print(f"\n{'*' * 85}\n[PRE-PROCESS] 의미 확장 쿼리: {expanded_query}\n{'*' * 85}")

    llm = ChatOllama(model=MODEL_NAME, temperature=0.02, streaming=True)

    # [2단계] 문서 검색
    retrieved_docs = rag_engine.compression_retriever.invoke(expanded_query)
    all_candidates = retrieved_docs[:12] if retrieved_docs else []

    # [3단계] 답변 생성 (도입부 변수 없이 context만 전달)
    chain = create_stuff_documents_chain(llm, get_qa_prompt())
    full_response = ""
    
    async for chunk in chain.astream({
        "input": query, 
        "chat_history": history.messages, 
        "context": all_candidates
    }):
        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
        full_response += chunk

    # [4단계] 후행 검증 및 터미널 로그 출력
    temp_sources = []
    seen = set()
    STRICT_GAP = 0.05
    
    densities = [calculate_fuzzy_density(d.page_content, full_response) for d in all_candidates]
    absolute_max_density = max(densities) if densities else 0

   # 터미널에 로그 출력 (기존 유지)
    print(f"\n[STEP 2] 후행 검증 (Fuzzy MAX Density: {absolute_max_density:.4f})")
    print(f"{'-'*85}")

    if "확인불가" not in full_response:
        # 💡 핵심 수정: 밀도가 0이더라도 LLM이 답변을 생성했다면 
        # (즉, 임베딩 검색은 성공해서 context에 데이터가 있었다면)
        # 1순위 문서는 일단 PASS 시켜서 출처가 누락되지 않게 합니다.
        
        if absolute_max_density > 0:
            best_idx = densities.index(absolute_max_density)
        else:
            best_idx = 0 # 밀도가 0이어도 검색 1순위 문서를 기준으로 잡음
            
        best_d = all_candidates[best_idx]
        f_name = os.path.basename(best_d.metadata.get('source', '문서'))
        f_page = best_d.metadata.get('page', 0) + 1
        
        source_text = f"\n\n---\n💡 자세한 정보는 [{f_name}, p.{f_page}]에서 추가적으로 더 찾아보세요."
        yield f"data: {json.dumps({'type': 'content', 'delta': source_text})}\n\n"

        for i, (d, match_density) in enumerate(zip(all_candidates, densities), 1):
            f = os.path.basename(d.metadata.get('source', 'unknown'))
            p = d.metadata.get('page', 0) + 1
            
            # 밀도가 0이어도 1순위거나, 최고 밀도와 차이가 적으면 PASS
            is_relevant = (absolute_max_density - match_density <= STRICT_GAP) or (i == 1 and absolute_max_density == 0)
            
            status = "✅ PASS" if is_relevant else "❌ DROP"
            print(f"{status:^8} | {f[:30]:<35} (p.{p}) | 밀도:{match_density:.4f}")

            if is_relevant and f"{f}_{p}" not in seen:
                temp_sources.append({"file": f, "page": p, "snippet": d.page_content[:150].strip() + "..."})
                seen.add(f"{f}_{p}")

    yield f"data: {json.dumps({'type': 'sources', 'data': temp_sources})}\n\n"
    yield "data: [DONE]\n\n"