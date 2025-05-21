"""
추천 엔진 모듈

이 모듈은 키워드 기반 장소 추천 기능을 제공합니다.
주요 구성요소:
    - RecommendationEngine: 추천 엔진 클래스
"""

import math
from app.logging.config import get_logger

from typing import Dict, List, Any
from collections import defaultdict
from app.services.vector_store import PlaceStore
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
    
    def get_recommendations(
        self,
        keywords: Dict[str, List[str]]
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
            self.logger.info(f"추천 시작 : 키워드={keywords}")
            category_place_max_scores = defaultdict(lambda: defaultdict(float))
            
            # 전체 키워드 수를 고려한 카테고리 가중치 설정
            total_keywords_num = sum(len(v) for v in keywords.values())
            category_weight = (total_keywords_num//2) + 1

            # 최종 추천 장소 유사도 임계치 설정
            SIMILARITY_THRESHOLD = category_weight * 0.8

            # 각 카테고리별로 처리
            for category, keyword_list in keywords.items():
                if not keyword_list:  # 키워드가 없는 카테고리는 건너뛰기
                    # self.logger.info(f"카테고리 '{category}'에 키워드가 없어 건너뜁니다.")
                    continue
                    
                # self.logger.info(f"카테고리 '{category}' 처리 중")
                for keyword in keyword_list:
                    if not keyword:  # 빈 키워드는 건너뛰기
                        continue
                        
                    # self.logger.info(f"키워드 '{keyword}' 검색 중")
                    try:
                        # 장소 검색
                        results, keyword_emb = self.place_store.search_places(category, keyword)
                        if not results or not results.get('metadatas') or not results['metadatas'][0]:
                            # self.logger.warning(f"키워드 '{keyword}'에 대한 검색 결과가 없습니다.")
                            continue
                            
                        # 유사도 계산 및 점수 누적
                        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                            if meta is None:
                                continue
                            if dist is None or (isinstance(dist, float) and math.isnan(dist)):
                                continue
                            pid = meta.get("place_id")
                            if not pid:
                                continue
                            sim = 1 - dist

                            category_place_max_scores[category][pid] = max(category_place_max_scores[category][pid], sim)

                    except Exception as e:
                        self.logger.error(f"키워드 '{keyword}' 처리 중 오류 발생: {str(e)}")
                        continue
            
            final_scores = defaultdict(float)
            for category, place_dict in category_place_max_scores.items():
                for pid, score in place_dict.items():
                    if category == "장소 카테고리":
                        score *= category_weight
                    final_scores[pid] += score

            if not final_scores:
                # self.logger.warning("추천할 장소가 없습니다.")
                return RecommendResponse(recommendations=[])
            
            # 필터링 및 정렬
            filtered_scores = {pid: score for pid, score in final_scores.items() if score >= SIMILARITY_THRESHOLD}
            sorted_places = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)

            # self.logger.info(f"최종 추천 장소: {sorted_places}")
            
            recommendations = [
                Recommendation(
                    id=pid,
                    similarity_score=score
                )
                for pid, score in sorted_places
            ]
            
            return RecommendResponse(recommendations=recommendations)
            
        except Exception as e:
            self.logger.error(f"추천 생성 중 오류 발생: {str(e)}")
            raise Exception(f"추천 생성 중 오류 발생: {str(e)}") 