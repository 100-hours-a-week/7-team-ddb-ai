import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import sqlite3

import io
import time
import boto3
import chromadb

from sqlite3 import OperationalError
from chromadb.errors import IDAlreadyExistsError

from app.core.config import settings
from app.core.constants import CATEGORY_MAP


def upload_chromadb(place_table, keywords, embedding_model):
    client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)

    for category, keyword_list in keywords.items():
        if not keyword_list:
            continue

        collection_name = CATEGORY_MAP.get(category)
        if not collection_name:
            continue

        collection = client.get_collection(name=collection_name)

        for keyword in keyword_list:
            try:
                doc_id = f"{collection_name}_{place_table['id'][0]}_{keyword}"
                metadata = {
                    "place_id": doc_id,
                    "keyword": keyword,
                    "category": category
                }
            except Exception as e:
                print(f"⚠️ 메타데이터 정제 실패: {e}")
                continue

            # 중복 방지
            try:
                existing = collection.get(ids=[doc_id])
                if existing["ids"]:
                    print(f"⚠️ 중복된 ID 감지됨: {doc_id}")
                    continue
            except Exception as e:
                print(f"⚠️ 중복 검사 실패 ({doc_id}): {e}")
                continue

            # 임베딩 생성
            try:
                keyword_vec = embedding_model.encode(keyword).tolist()
            except Exception as e:
                print(f"❌ 임베딩 실패 ({keyword}): {e}")
                continue

            # 데이터 추가 5회 반복
            for attempt in range(5):
                try:
                    collection.add(
                        ids=[doc_id],
                        documents=[keyword],
                        metadatas=[metadata],
                        embeddings=[keyword_vec]
                    )
                    break  # 성공 시 반복 탈출
                except (OperationalError, IDAlreadyExistsError) as e:
                    print(f"⏳ ({attempt+1}/5) 잠금 또는 중복 오류 발생: {e}")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"❌ 알 수 없는 오류 발생 ({keyword}): {e}")
                    break  # 비재시도 오류
            else:
                print(f"❌ 최대 재시도 초과, 업로드 실패: {doc_id}")


def upload_s3(place_table, place_hours_table, place_menu_table):
    s3 = boto3.client(
    's3',
    aws_access_key_id=settings.S3_ACCESS_KEY,
    aws_secret_access_key=settings.S3_SECRET_KEY,
    region_name=settings.S3_DEFAULT_REGION
    )

    place_id = place_table['id'][0]

    # -------- place_table.csv 업로드 --------
    # CSV 데이터를 메모리에 쓰기
    place_table_csv_buffer = io.StringIO()
    place_table.to_csv(place_table_csv_buffer, index=False, encoding="utf-8-sig")
    place_table_key = f"{place_id}/place_table.csv"

    # S3에 업로드
    s3.put_object(
        Bucket=settings.S3_METADATA_PATH,
        Key=place_table_key,
        Body=place_table_csv_buffer.getvalue(),
        ContentType="text/csv"
    )

    # -------- place_hours_table.csv 업로드 --------
    place_hours_table_csv_buffer = io.StringIO()
    place_hours_table.to_csv(place_hours_table_csv_buffer, index=False, encoding="utf-8-sig")
    place_hours_table_key = f"{place_id}/place_hours_table.csv"

    s3.put_object(
        Bucket=settings.S3_METADATA_PATH,
        Key=place_hours_table_key,
        Body=place_hours_table_csv_buffer.getvalue(),
        ContentType="text/csv"
    )

    # -------- place_menu_table.csv 업로드 --------
    place_menu_table_csv_buffer = io.StringIO()
    place_menu_table.to_csv(place_menu_table_csv_buffer, index=False, encoding="utf-8-sig")
    place_menu_table_key = f"{place_id}/place_menu_table.csv"

    s3.put_object(
        Bucket=settings.S3_METADATA_PATH,
        Key=place_menu_table_key,
        Body=place_menu_table_csv_buffer.getvalue(),
        ContentType="text/csv"
    )

    # -------- 이미지 존재하면 이미지 업로드 --------
    image_url = place_table['image_url'][0]
    if image_url:
        temp_image_path = f"{settings.TEMP_IMAGE_PATH}/{place_id}.jpg"
        image_key = f"place/{place_id}.jpg"

        s3.upload_file(
            Filename=temp_image_path,
            Bucket=settings.S3_IMAGE_PATH,
            Key=image_key,
            ExtraArgs={"ContentType": "image/jpeg"}
        )