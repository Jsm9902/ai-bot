import json
import os
import re
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from config import MODEL_NAME
from engine import rag_engine
from prompts import get_qa_prompt, QUERY_EXPANSION_PROMPT

# [개선] 조사 제거 및 정규화를 통해 한국어 매칭 밀도를 계산하는 함수
def calculate_flexible_density(target_content, search_text):
    if not search_text: return 0.0
    clean_target = re.sub(r'\s+', '', target_content).lower()
    raw_words = re.split(r'\s+', search_text)
    # 조사(의, 이, 가 등)를 제거하여 핵심 키워드 위주로 추출
    words = [re.sub(r'(의|이|가|을|를|은|는|야|니|\?)$', '', w) for w in raw_words if len(w) >= 1]
    if not words: return 0.0
    match_count = sum(1 for w in words if w.lower() in clean_target)
    return match_count / len(words)

async def stream_answer(query: str, history):
    if rag_engine.compression_retriever is None:
        rag_engine.setup_engine()

    # [1단계] 쿼리 확장
    expansion_llm = ChatOllama(model=MODEL_NAME, temperature=0)
    expansion_prompt = PromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
    expansion_chain = expansion_prompt | expansion_llm
    expanded_result = await expansion_chain.ainvoke({"input": query})
    expanded_query = expanded_result.content.strip()

    print(f"\n{'*' * 85}\n[PRE-PROCESS] 확장 쿼리: {expanded_query}\n{'*' * 85}")

    # [2단계] 문서 검색
    retrieved_docs = rag_engine.compression_retriever.invoke(expanded_query)
    all_candidates = retrieved_docs[:7] if retrieved_docs else []

    # 💡 [DEBUG] 후보 문서 리스트 출력
    print(f"\n[DEBUG] 후보 문서 리스트 (Top-7):")
    for i, doc in enumerate(all_candidates, 1):
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 0) + 1
        print(f"   {i}. {source} (p.{page})")

    # 💡 [STEP 3] 사전 검증 및 고순도 문서 선별 (STRICT_GAP 적용)
    print(f"\n[STEP 3] 강제 보정 검증 및 밀도 체크")
    print("-" * 85)
    
    densities = []
    for d in all_candidates:
        densities.append(calculate_flexible_density(d.page_content, query))

    absolute_max_density = max(densities) if densities else 0
    STRICT_GAP = 0.05 # 1순위 문서와 성격이 다른 문서를 쳐내기 위한 격차 기준
    
    passed_docs = [] # 👈 실제로 LLM 답변 생성에 사용될 정제된 문서 리스트
    final_densities = [] # PASS된 문서들의 밀도 리스트

    for i, (d, density) in enumerate(zip(all_candidates, densities)):
        source_name = os.path.basename(d.metadata.get('source', 'unknown'))
        page_info = f"(p.{d.metadata.get('page', 0) + 1})"
        
        # 1. 최소 밀도 0.1 이상
        # 2. 1순위 문서이거나, 최고 밀도와의 차이가 0.05 이내인 경우만 PASS
        is_relevant = (density >= 0.1) and (
            (i == 0) or (absolute_max_density - density <= STRICT_GAP)
        )
        
        status = "✅ PASS" if is_relevant else "❌ DROP"
        print(f" {status} | {source_name:<30} {page_info:<8} | 밀도:{density:.4f}")
        
        if is_relevant:
            passed_docs.append(d)
            final_densities.append(density)

    print("-" * 85)
    print(f"[STEP 3] 사전 검증 완료 (최종 PASS: {len(passed_docs)}개 / MAX Density: {absolute_max_density:.4f})")
    print("-" * 85)

    # 💡 [4단계] 답변 생성 전략 (선별된 passed_docs만 전달)
    modified_query = query
    if absolute_max_density >= 0.25:
        modified_query = (
            f"제공된 문서에 질문과 관련된 핵심 키워드가 충분히 포함되어 있습니다. "
            f"절대로 '확인불가'라고 답하지 말고, 문서 내용을 바탕으로 상세히 답변하세요.\n"
            f"질문: {query}"
        )

    llm = ChatOllama(model=MODEL_NAME, temperature=0.1, streaming=True) 
    chain = create_stuff_documents_chain(llm, get_qa_prompt())
    full_response = ""

    # context에 all_candidates 대신 passed_docs를 넣어 정보 오염 방지
    async for chunk in chain.astream({
        "input": modified_query,
        "chat_history": history.messages,
        "context": passed_docs 
    }):
        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
        full_response += chunk

    # [5단계] 사후 개입 (AI가 실패했을 때 안전장치)
    is_failed_answer = "확인불가" in full_response
    temp_sources = []
    seen = set()

    if is_failed_answer and absolute_max_density >= 0.2:
        override_msg = "\n\n---\n💡 **참고: AI가 직접적인 답변 생성을 거부했으나, 문서 내 관련 데이터는 다음과 같습니다.**\n"
        yield f"data: {json.dumps({'type': 'content', 'delta': override_msg})}\n\n"

        for d, density in zip(passed_docs, final_densities):
            f_name = os.path.basename(d.metadata.get('source', 'unknown'))
            f_page = d.metadata.get('page', 0) + 1
            snippet = d.page_content[:200].strip().replace('\n', ' ')
            
            source_text = f"📍 **[{f_name}, p.{f_page}]**\n> {snippet}...\n\n"
            yield f"data: {json.dumps({'type': 'content', 'delta': source_text})}\n\n"
            
            if f"{f_name}_{f_page}" not in seen:
                temp_sources.append({"file": f_name, "page": f_page, "snippet": snippet})
                seen.add(f"{f_name}_{f_page}")
    else:
        # 정상 답변 시에도 선별된 passed_docs에서만 출처 추출
        for d in passed_docs:
            f = os.path.basename(d.metadata.get('source', 'unknown'))
            p = d.metadata.get('page', 0) + 1
            if f"{f}_{p}" not in seen:
                temp_sources.append({"file": f, "page": p, "snippet": d.page_content[:150].strip()})
                seen.add(f"{f}_{p}")

    if temp_sources:
        yield f"data: {json.dumps({'type': 'sources', 'data': temp_sources})}\n\n"
    yield "data: [DONE]\n\n"