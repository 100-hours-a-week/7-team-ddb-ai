"""
추천 엔진 모듈

이 모듈은 키워드 기반 장소 추천 기능을 제공합니다.
주요 구성요소:
    - RecommendationEngine: 추천 엔진 클래스
"""

import asyncio
import numpy as np
import pandas as pd

from typing import Optional, List, Tuple
from collections import defaultdict
from app.logging.config import get_logger
from app.services.recommend.retriever import PlaceStore
from app.schemas.recommend_schema import Recommendation, RecommendResponse

class RecommendationEngine:
    """
    추천 엔진 클래스
    
    이 클래스는 키워드 기반으로 장소를 추천합니다.
    
    Attributes:
        place_store (PlaceStore): 장소 벡터 저장소
    """
    
    def __init__(self, place_store: PlaceStore, logger=None):
        """
        RecommendationEngine 초기화
        
        Args:
            place_store (PlaceStore): 장소 벡터 저장소 인스턴스
        """
        self.place_store = place_store
        if logger is None:
            from app.logging.di import get_logger_dep
            logger = get_logger_dep()
        self.logger = logger
    
    async def get_recommendations(
        self,
        categories: Optional[List[str]],
        keyword_vecs: Optional[List[float]],
        place_category: Optional[str]
    ) -> RecommendResponse:
        """
        키워드 기반 장소 추천
        
        Args:
            keywords (Dict[str, List[str]]): 카테고리별 키워드 목록
            top_n (int): 반환할 추천 장소 수
            
        Returns:
            RecommendResponse: 추천 결과
            
        Raises:
            Exception: 추천 생성 중 오류 발생 시
        """
        try:
            keyword_weight, place_threshold = self._calculate_weight_threshold(categories)
            place_scores = await self._calculate_place_scores(categories, keyword_vecs, keyword_weight)
            filtered_df = self._filter_and_sort(place_scores, place_threshold)

            recommendations = [
                Recommendation(
                    id=row.place_id,
                    similarity_score=row.total_score,
                    keyword=row.keywords,
                )
                for _, row in filtered_df.iterrows()
            ]

            return RecommendResponse(
                recommendations=recommendations,
                place_category=place_category,
            )
        
        except Exception as e:
            self.logger.error(f"추천 생성 중 오류 발생: {str(e)}")
            raise Exception(f"추천 생성 중 오류 발생: {str(e)}") 

    @staticmethod
    def _calculate_weight_threshold(categories: List[str]) -> Tuple[float, float]:
        """
        키워드 비율에 따른 음식/제품 키워드 가중치, 추천 장소 임계값 계산
        """
        total = len(categories)

        food_product_count = categories.count("음식/제품")
        keyword_weight = total - food_product_count + 1

        place_threshold = ((food_product_count *  keyword_weight) + (total - food_product_count)) * 0.7

        return keyword_weight, place_threshold

    async def _calculate_place_scores(
        self,
        categories: List[str],
        keyword_vecs: List[float],
        keyword_weight: float
    ) -> dict[int, dict[str, object]]:
        """
        장소 별 가장 유사한 키워드들의 유사도 누적
        """
        place_scores = defaultdict(lambda: {"total_score": 0.0, "keywords": set()})

        for category, keyword_vec in zip(categories, keyword_vecs):
            try:
                results = await asyncio.to_thread(self.place_store.search_places, category, keyword_vec)
                if not results:
                    continue

                self._best_place_scores(results, category, place_scores, keyword_weight)

            except Exception as e:  # pragma: no cover
                self.logger.error("카테고리 %s 처리 중 오류: %s", category, e, exc_info=True)

        return place_scores

    def _best_place_scores(
        self,
        results: dict,
        category: str,
        place_scores: dict[int, dict[str, object]],
        keyword_weight: float
    ):
        """
        검색된 유사 키워드에서 장소 중복을 제거하여 한 장소 별 가장 유사한 키워드만 필터링
        """
        metas = results["metadatas"][0]
        scores = self._convert_distance_to_score(np.array(results["distances"][0]), category, keyword_weight)

        best_scores = {}
        for meta, score in zip(metas, scores):
            pid = meta.get("place_id")
            kw = meta.get("keyword")
            if pid is None:
                continue
            # 한 장소-한 키워드 최고 점수만 유지
            if pid not in best_scores or best_scores[pid]["score"] < score:
                best_scores[pid] = {"score": score, "keyword": kw}

        # 점수 누적
        for pid, info in best_scores.items():
            place_scores[pid]["total_score"] += info["score"]
            place_scores[pid]["keywords"].add(info["keyword"])

    @staticmethod
    def _convert_distance_to_score(distances: np.ndarray, category: str, keyword_weight: float) -> np.ndarray:
        """
        거리 → 유사도 점수 변환 (카테고리 가중치 포함)
        """
        scores = 1.0 - distances
        if category == "음식/제품":  # 가중치 부여
            scores *= keyword_weight
        return scores

    def _filter_and_sort(
        self,
        place_scores: dict[int, dict[str, object]],
        place_threshold: float
    ) -> pd.DataFrame:
        """
        내부 임계값 기준 필터링 후 점수 내림차순 정렬
        """
        df = pd.DataFrame(
            {
                "place_id": pid,
                "total_score": data["total_score"],
                "keywords": list(data["keywords"]),
            }
            for pid, data in place_scores.items()
        )

        return df[df["total_score"] >= place_threshold].sort_values("total_score", ascending=False)