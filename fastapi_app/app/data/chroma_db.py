import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import sqlite3

import os
import json
import chromadb
import pandas as pd

from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.core.config import settings
from app.core.constants import CATEGORY_MAP

def is_valid_embedding(vec, expected_dim=768):
    if not isinstance(vec, list):
        return False
    if len(vec) != expected_dim:
        return False
    if not all(isinstance(x, float) for x in vec):
        return False
    if any(x != x or abs(x) == float("inf") for x in vec):
        return False
    return True

def get_hnsw_metadata_by_size(size: int) -> dict:
    if size < 333:
        return {
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:search_ef": 100
        }
    elif size < 666:
        return {
            "hnsw:space": "cosine",
            "hnsw:M": 24,
            "hnsw:search_ef": 150
        }
    else:
        return {
            "hnsw:space": "cosine",
            "hnsw:M": 32,
            "hnsw:search_ef": 200
        }

def make_chroma_db():
    # 기본 파일 경로
    csv_path = "app/data/place_id_category_data.csv"
    jsonl_path = "app/data/place_keywords.jsonl"
    chroma_path = settings.VECTOR_STORE_PATH

    # ✅ 임베딩 모델 설정
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    # ✅ Chroma 저장 경로 생성
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    embedding_func = SentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL_NAME)

    # ✅ 데이터 로드
    df_place_ids = pd.read_csv(csv_path)
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        place_keyword_data = [json.loads(line) for line in f]

    print(f"총 {len(place_keyword_data)}개의 장소 키워드 데이터 로드됨.")

    for entry in place_keyword_data:
        place_name = entry["place_name"]
        keywords_by_category = entry["keywords"]

        row = df_place_ids[df_place_ids["name"] == place_name]
        if row.empty:
            print(f"❗ place_id 누락 - '{place_name}'")
            continue
        
        place_id = int(row["id"].values[0])
        place_category = row["place_category"].values[0]

        # ✅ '음식/제품' 키가 없으면 새로 리스트 만들고 추가
        if "음식/제품" not in keywords_by_category:
            keywords_by_category["음식/제품"] = []

        # ✅ 카테고리 값이 null이 아니면 추가
        if pd.notnull(place_category):
            keywords_by_category["음식/제품"].append(place_category)

        for kor_category, keyword_list in keywords_by_category.items():
            if not keyword_list:
                continue

            collection_name = CATEGORY_MAP.get(kor_category)
            if not collection_name:
                continue

            try:
                collection = client.get_collection(name=collection_name)
            except:
                collection_size = len(keyword_list)
                metadata = get_hnsw_metadata_by_size(collection_size)

                collection = client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_func,
                    metadata=metadata
                )

            for keyword in keyword_list:
                try:
                    vec = embedding_model.encode(keyword).tolist()

                    if not is_valid_embedding(vec, expected_dim=embedding_dim):
                        print(f"❌ 유효하지 않은 임베딩: {keyword}")
                        continue

                    doc_id = f"{collection_name}_{place_id}_{keyword}"
                    metadata = {
                        "place_id": place_id,
                        "keyword": keyword,
                        "category": kor_category
                    }

                    # 중복 방지
                    existing = collection.get(ids=[doc_id])
                    if existing["ids"]:
                        print(f"⚠️ 중복된 ID 감지됨: {doc_id}")
                        continue

                    collection.add(
                        ids=[doc_id],
                        documents=[keyword],
                        metadatas=[metadata],
                        embeddings=[vec]
                    )

                except Exception as e:
                    print(f"❌ 오류 발생 ({keyword}): {e}")