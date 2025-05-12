import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import sqlite3

import os
import json
import chromadb
import pandas as pd

from tqdm import tqdm
from app.core.constants import CATEGORY_MAP
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ✅ 임베딩 모델 설정 (KRSBERT STS)
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
embedding_dim = embedding_model.get_sentence_embedding_dimension()

# ✅ Chroma 저장 경로
chroma_path = "chroma_db_keyword"
os.makedirs(chroma_path, exist_ok=True)

client = chromadb.PersistentClient(path=chroma_path)
embedding_func = SentenceTransformerEmbeddingFunction(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

df_place_ids = pd.read_csv("more_data_for_postgres.csv")

# ✅ 키워드 JSON 로드
place_keyword_file = "more_place_outputs.jsonl"
with open(place_keyword_file, "r", encoding="utf-8-sig") as f:
    place_keyword_data = [json.loads(line) for line in f]

print(len(place_keyword_data))

# ✅ 유효한 임베딩인지 검사하는 함수
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

for entry in tqdm(place_keyword_data):
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

    # ✅ 값이 null이 아니고 문자열이면 추가
    if pd.notnull(place_category):
        keywords_by_category["음식/제품"].append(place_category)
        
    for kor_category, keyword_list in keywords_by_category.items():
        if not keyword_list:
            continue

        collection_name = CATEGORY_MAP.get(kor_category)
        if not collection_name:
            continue

        collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_func)

        for keyword in keyword_list:
            try:
                # ✅ 키워드 임베딩 수동 생성
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

                # 생성 직전에 동일 ID 존재하는지 확인
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
