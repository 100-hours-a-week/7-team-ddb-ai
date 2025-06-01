# Dolpin AI - 장소 추천 시스템

## 프로젝트 개요
Dolpin AI는 사용자의 자연어 입력을 기반으로 장소를 추천하는 AI 시스템입니다. LLM을 활용하여 키워드를 추출하고, 벡터 데이터베이스를 통해 유사한 장소를 찾아 추천합니다.

## 주요 기능
- 자연어 기반 장소 추천
- LLM을 활용한 키워드 추출
- 벡터 데이터베이스 기반 유사도 검색
- 실시간 추천 처리

## 기술 스택
- FastAPI: 백엔드 프레임워크
- LangChain: LLM 통합
- Chroma: 벡터 데이터베이스
- Docker: 컨테이너화
- Prometheus/Grafana: 모니터링

## 프로젝트 구조
```
fastapi_app/
├── app/
│   ├── main.py                      # FastAPI 진입점
│   ├── api/                         # 엔드포인트 및 라우터
│   │   ├── deps.py                  # Depends 의존성
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   └── recommend.py     # 추천 요청 엔드포인트
│   │       └── router.py            # v1 router 통합
│   ├── services/                    # 서비스/비즈니스 로직
│   │   ├── recommender.py           # 추천 서비스
│   │   ├── recommend_engine.py      # 추천 엔진
│   │   ├── vector_store.py          # 벡터DB 연동
│   ├── schemas/                     # Pydantic 데이터 모델
│   │   └── recommend_schema.py      # 입력/출력 스키마
│   ├── core/                        # 설정, 상수, 초기화
│   │   ├── config.py                # 기본 설정
│   │   ├── constants.py             # 상수
│   │   └── init_app.py              # 앱 부트스트랩
│   ├── cache/                       # 캐싱
│   │   └── redis.py                 # Redis 연동
│   ├── data/                        # 데이터 파일
│   │   ├── place_keywords.jsonl     # 장소 키워드 데이터
│   │   ├── place_id_category_data.csv # 장소-카테고리 매핑
│   │   ├── chroma_db.py             # ChromaDB 유틸
│   │   └── chroma_db_keyword/       # ChromaDB 키워드 데이터
│   ├── database/                    # DB 설정
│   ├── logging/                     # 로깅
│   │   ├── config.py                # 로깅 설정
│   │   └── di.py                    # DI 유틸
├── config/                          # 환경 설정
│   ├── development.py               # 개발 환경
│   ├── production.py                # 운영 환경
│   └── testing.py                   # 테스트 환경
├── docs/                            # 문서화
│   └── deployment/                  # 배포 문서
├── monitoring/                      # 모니터링
│   └── metrics.py                   # 메트릭 수집
├── scripts/                         # 유틸리티 스크립트
│   ├── deploy.sh                    # 배포 스크립트
│   └── setup.sh                     # 환경 설정 스크립트
├── tests/                           # 테스트
│   ├── e2e/                         # E2E 테스트
│   ├── integration/                 # 통합 테스트
│   └── unit/                        # 단위 테스트
├── requirements.txt                 # Python 의존성 목록
├── Dockerfile                       # 컨테이너 설정
├── README.md                        # 프로젝트 문서
└── pysqlite3_binary-...whl          # (옵션) 수동 wheel 파일
```

1. **핵심 앱 코드(app/)**
    - `main.py`: FastAPI 앱 진입점
    - `api/`: 엔드포인트, 라우터, 의존성 관리
    - `services/`: 추천, 벡터DB 등 서비스 로직
    - `schemas/`: Pydantic 기반 데이터 모델(입출력 스키마)
    - `core/`: 환경설정, 상수, 앱 초기화
    - `cache/`: Redis 등 캐싱 관련 코드
    - `data/`: 데이터 파일 및 ChromaDB 유틸
    - `database/`: DB 설정
    - `logging/`: 로깅 설정

2. **환경 설정(config/)**  
    - 개발/운영/테스트 환경별 설정 파일

3. **문서화(docs/)**  
    - 배포 등 프로젝트 문서

4. **모니터링(monitoring/)**  
    - 메트릭 수집 등 모니터링 코드

5. **유틸리티 스크립트(scripts/)**  
    - 배포, 환경설정 등 각종 스크립트

6. **테스트(tests/)**  
    - 단위/통합/E2E 테스트 코드

7. **의존성 및 배포**
    - `requirements.txt`: Python 패키지 의존성 목록
    - `Dockerfile`: 컨테이너 이미지 정의
    - `README.md`: 프로젝트 설명서
    - `pysqlite3_binary-...whl`: (필요시) 수동 wheel 파일

## 설치 및 실행

### 요구사항
- Python 3.8+
- Docker
- Chroma

### 설치
```bash
# 저장소 클론
git clone https://github.com/100-hours-a-week/7-team-ddb-ai.git

# 의존성 설치
pip install -r requirements.txt
```

### 실행
```bash
cd fastapi_app
# 개발 환경
uvicorn app.main:app --reload

# Docker 환경
docker-compose up -d
```

## API 사용법

### 추천 API
```http
POST /api/v1/recommend
Content-Type: application/json

{
    "text": "커피가 맛있는 카공하기 좋은 카페 추천해줘"
}
```

응답:
```json
{
  "recommendations": [
    {
      "id": 18612362,
      "similarity_score": 4.304481685161591,
      "keyword": [
        "카페 아메리카노",
        "분위기 좋음",
        "카페"
      ]
    },
    {
      "id": 1918499280,
      "similarity_score": 4.203140914440155,
      "keyword": [
        "더치커피",
        "좋은 분위기",
        "카페"
      ]
    }
  ]
}
```

## 모니터링
- Prometheus: 메트릭 수집
- Grafana: 대시보드 및 시각화
- 로깅: 구조화된 로그 수집


## contribution
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request