import json
import os
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import MODEL_NAME
from engine import rag_engine
from prompts import get_qa_prompt

async def stream_answer(query: str, history):
    if rag_engine.compression_retriever is None:
        rag_engine.setup_engine()
    
    if not rag_engine.compression_retriever:
        yield f"data: {json.dumps({'type': 'content', 'delta': '업로드된 문서가 없습니다.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    llm = ChatOllama(model=MODEL_NAME, temperature=0, streaming=True)

    # 1. 문서 검색
    retrieved_docs = rag_engine.compression_retriever.invoke(query)
    
    def extract_score(doc):
        score = doc.metadata.get('relevance_score') or doc.metadata.get('score') or getattr(doc, 'score', None)
        try: return float(score) if score is not None else 0.0
        except: return 0.0

    # 2. 지능형 다중 파일 필터링 로직
    filtered_docs = []
    print(f"\n--- [DEBUG] 실시간 검색 분석 ---")
    
    if retrieved_docs:
        # 1등 문서는 기준점이므로 무조건 포함
        first_doc = retrieved_docs[0]
        first_file = os.path.basename(first_doc.metadata.get('source', ''))
        filtered_docs.append(first_doc)
        print(f"1순위(기준): {first_file} | 점수: {extract_score(first_doc):.4f}")

        # 상위 후보군(2~4위) 검토
        for i, doc in enumerate(retrieved_docs[1:4], start=2):
            score = extract_score(doc)
            curr_file = os.path.basename(doc.metadata.get('source', ''))
            
            # [통과 조건]
            # 1. 점수가 있다면 1등의 60% 이상일 때 (다른 파일이라도 OK)
            if score > 0 and score >= (extract_score(first_doc) * 0.6):
                filtered_docs.append(doc)
                print(f"{i}순위(점수통과): {curr_file}")
            # 2. 점수가 없어도 2순위라면 일단 포함 (다중 파일 지원)
            elif len(filtered_docs) < 2:
                filtered_docs.append(doc)
                print(f"{i}순위(순위통과): {curr_file}")
            # 3. 그 외에는 1등 문서와 같은 파일일 때만 추가 (노이즈 방지)
            elif curr_file == first_file:
                filtered_docs.append(doc)
                print(f"{i}순위(출처일치): {curr_file}")

    if not filtered_docs:
        yield f"data: {json.dumps({'type': 'content', 'delta': '문서에서 확인 불가'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 3. LLM 실행
    chain = create_stuff_documents_chain(llm, get_qa_prompt())
    full_response = ""
    
    async for chunk in chain.astream({
        "input": query, 
        "chat_history": history.messages, 
        "context": filtered_docs
    }):
        yield f"data: {json.dumps({'type': 'content', 'delta': chunk})}\n\n"
        full_response += chunk

    # 4. 출처 표시 및 사후 검증
    stop_keywords = ["확인 불가", "정보 없음", "내용 없음", "찾을 수 없습니다"]
    if not any(keyword in full_response for keyword in stop_keywords):
        seen = set()
        sources = []
        for d in filtered_docs:
            p = d.metadata.get('page', 0) + 1
            f = os.path.basename(d.metadata.get('source', ''))
            if f"{f}_{p}" not in seen:
                sources.append({
                    "file": f, "page": p, 
                    "snippet": d.page_content[:150] + "..."
                })
                seen.add(f"{f}_{p}")
        
        sources.sort(key=lambda x: (x['file'], x['page']))
        history.add_user_message(query)
        history.add_ai_message(full_response)
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
    
    yield "data: [DONE]\n\n"