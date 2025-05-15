## 🛠 배포 전략 개요 (Deployment Overview)

우리 팀의 배포 파이프라인은 초기에는 GitHub 브랜치 전략과 GCP Bastion 서버를 활용한 수동 배포에서 시작하여, 이후 점차적으로 CI/CD 자동화, 컨테이너화, Kubernetes 오케스트레이션, 배포 자동화 도구까지 확장되는 구조를 가집니다.

각 단계는 실 운영 경험과 기술 도입 시점을 기준으로 문서화되며, 단계별로 독립적인 파일에 정리되어 있습니다.

### 📚 문서 목차

1. [1단계 - Git 브랜치 기반 + Bastion 수동 배포](01_git-bastion-manual.md)
2. [2단계 - Jenkins + Terraform을 통한 배포 자동화](02_jenkins-terraform.md)
3. [3단계 - Docker 기반 컨테이너화 배포](03_docker-deployment.md)
4. [4단계 - kubeadm을 활용한 오케스트레이션](04_kubeadm-orchestration.md)
5. [5단계 - k8s를 활용한 배포 자동화 및 모니터링](05_k8s-automation.md)
