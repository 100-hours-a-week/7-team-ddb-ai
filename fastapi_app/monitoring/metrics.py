from prometheus_client import Counter, Histogram

# 추천 API 관련 메트릭을 관리하는 클래스
class RecommendMetrics:
    def __init__(self):
        # 추천 API 요청 수를 카운트하는 Counter 메트릭
        self.request_count = Counter(
            'recommend_requests_total', '추천 API 요청 수'
        )
        # 추천 API 요청 처리 시간을 기록하는 Histogram 메트릭
        self.request_latency = Histogram(
            'recommend_request_latency_seconds', '추천 API 요청 처리 시간'
        )

# RecommendMetrics의 싱글턴 인스턴스 생성 (프로젝트 전체에서 공유)
metrics = RecommendMetrics()  # 싱글턴 인스턴스 