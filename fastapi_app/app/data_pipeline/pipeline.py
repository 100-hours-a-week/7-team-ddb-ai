from app.data_pipeline.crawler import crawling
from app.data_pipeline.post_processor import post_processing
from app.data_pipeline.uploader import upload_chromadb
from app.data_pipeline.uploader import upload_s3
from app.services.recommend.embedding import EmbeddingModel

class UploaderPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel
    ):
        self.embedding_model = embedding_model

    def upload_data(self, place_id: int) -> None:
        place_table, place_hours_table, place_facilities, place_menu_table, place_reviews = crawling(place_id)
        place_table, keywords = post_processing(place_table, place_menu_table, place_facilities, place_reviews)
        upload_chromadb(place_table, keywords, self.embedding_model)
        upload_s3(place_table, place_hours_table, place_menu_table)