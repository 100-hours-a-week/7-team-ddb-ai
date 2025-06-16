# Python 3.10 slim 이미지를 베이스로 사용
FROM python:3.10-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y build-essential libsqlite3-dev && rm -rf /var/lib/apt/lists/*

# 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt ./
COPY pysqlite3_binary-0.5.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl .
COPY model /app/model

# wheel 설치에 필요한 최소 구성
RUN pip install --upgrade pip

RUN pip install ./pysqlite3_binary-0.5.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 전체 복사
COPY . .

# FastAPI 서버 실행 (main.py의 app 객체 기준)
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--no-access-log"]