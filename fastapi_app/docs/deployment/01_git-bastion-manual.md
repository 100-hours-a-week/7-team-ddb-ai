# 01. Git 브랜치 기반 + Bastion 수동 배포

## 1. 개요

현재 팀은 GitHub 브랜치를 활용한 버전 관리 및 배포 흐름을 운영하고 있으며, GCP에서 생성되는 Bastion 서버를 통해 AI 서버에 접속하여 직접 배포를 수행하는 구조를 사용하고 있습니다.

- 배포 환경: GCP 인스턴스
- 접근 방식: Bastion 서버 → AI 서버 (prod/dev 분리)
- 배포 방법: SSH 수동 접속 → git pull → 서버 프로세스 시작

## 2. 다이어그램

```scss
┌────────────────────┐
│     🧍 로컬 PC      │
│ (개발자, pem 보유)  │
└─────────┬──────────┘
          │ SSH (prod.pem / dev.pem)
          ▼
┌────────────────────────────┐
│   🛡️ Bastion 서버 (prod/dev)│  ← 매일 새로 생성됨 (IP, pem 공유됨)
│   (공인 IP, 외부 접근 가능)  │
└─────────┬──────────┬───────┘
          │          │
 SSH (고정 pem)      SSH (고정 pem)
          ▼          ▼
┌──────────────────────┐   ┌──────────────────────┐
│   🧠 AI 서버 (Prod)  │   │   🧪 AI 서버 (Dev)   │
│ - GCP 인스턴스        │   │ - GCP 인스턴스        │
│ - 내부망 IP or 고정 IP│   │ - 내부망 IP or 고정 IP│
│ - FastAPI 실행 중     │   │ - 개발 테스트 용도    │
└──────────────────────┘   └──────────────────────┘
```

## 3. Git 브랜치 전략

| 브랜치 | 설명 |
|--------|------|
| `main` | 운영용 브랜치 (prod 배포 기준) |
| `dev`  | 개발 서버 배포용 브랜치 |
| `feat/#`, `fix/#` | 기능 개발용 브랜치, PR을 통해 `dev`에 병합 |

- PR 기준으로 코드 리뷰 및 변경 이력 관리
- `dev`에서 충분히 테스트 후 `main`으로 머지하여 운영 반영

## 4. Bastion 서버를 통한 SSH 접속 흐름

```bash
# 1단계: Bastion 접속 (매일 pem, IP 갱신됨)
ssh -i <dev-bastion-key.pem> <your-username>@<dev-bastion-ip>
ssh -i <prod-bastion-key.pem> <your-username>@<prod-bastion-ip>

# 2단계: AI 서버 접속 (내부망 IP는 고정)
ssh -i <dev-ai-key.pem> <your-username>@<dev-ai-server-ip>
ssh -i <prod-ai-key.pem> <your-username>@<prod-ai-server-ip>
```

- 각 bastion 접속용 pem 키와 IP는 디스코드에서 매일 공유됨
- AI 서버는 고정 IP이나 외부 접근이 불가하여 반드시 bastion 경유 필요

## 5. 배포 절차 (FastAPI 기준)

```bash
# 1. 코드 최신화
cd ~/7-team-ddb-ai/fastapi_app
git pull

# 2. 서버 PID 확인
ps -ef | grep uvicorn

# 3. 서버 종료
kill <pid>

# 4. 재시작
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > output.log 2>&1 &

# 5. 로그 확인
cat output.log
```

- 운영(prod) 서버는 로그만 확인하고, 재시작은 금지
- 개발(dev) 서버는 위 절차에 따라 배포 가능
