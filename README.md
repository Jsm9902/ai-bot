해당 내용은 기존 ai bot에 ai가 어떤 문서를 참고하고 있는지
터미널에서 확인가능하도록 업그레이드 하였습니다.

1. 설치
공식 홈페이지 파이썬 3.12.9 설치
* py -3.12 --version (확인용)
2. visual Studio Build Tools 설치 (설치 후 다시시작)
* C++를 사용한 데스크톱 개발
* MSVC v14x - VS 2022 C++ x64/x86 빌드 도구 선택
* Windows 10(또는 11) SDK 선택
3. 공식 홈페이지에서 ollama 윈도우용 설치
4. llm 설치 (cmd)
* 답변 생성 모델 : ollama run hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4\_K\_M
* 임베딩 모델 : ollama pull mxbai-embed-large
5. 가상환경 실행 (프로젝트 폴더 위치에서 실행)
* py -3.12 -m venv venv
* .\\venv\\Scripts\\activate
* python -m pip install --upgrade pip (pip 최신 업데이트, 오류 방지용)
* pip install -r requirements.txt (패키지 설치)
6. 실행하기
* 프로젝트 폴더 위로 이동
* run.bat 입력
* 링크 복사 붙여넣기 후 들어가기
* 챗봇 사용 가능

