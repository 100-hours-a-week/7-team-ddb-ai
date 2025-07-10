import re
import json
import google.generativeai as genai

from google.generativeai import GenerativeModel

from app.core.config import settings


_model_instance = None


def get_prompt():
    prompt = """
    당신은 장소 키워드 추출 AI입니다.

    아래에 제시된 장소의 메타데이터에서 해당 장소의 특징을 잘 나타내는 키워드를 추출하세요.

    장소 메타데이터:
    {metadata_block}

    키워드는 다음 8개의 카테고리로 분류합니다:

    1. 음식/제품: 제품, 음식 키워드 혹은 해당 제품, 음식 대한 표현(예: '부드러운 고기', '싱싱한 해산물', '맛있는 커피', '삼겹살', '양고기')
    2. 분위기/공간: 인테리어, 조명, 소음, 청결 등 (예: '조용한 분위기', '깔끔한 공간')
    3. 서비스/직원: 직원 응대, 서비스 속도 등 (예: '친절함', '빠른 응대')
    4. 가격/가성비: 합리적 가격, 혜택 등 (예: '가격 만족도 높음', '가성비 좋음')
    5. 접근성/편의시설: 주차, 위치 접근성, 편의시설 등 (예: '넉넉한 주차', '역 근처 위치', '24시간', '와이파이')
    6. 방문 목적: 추천 대상, 상황 등 (예: '데이트 추천', '혼밥 추천')
    7. 장소 카테고리 : 장소를 분류할 수 있는 키워드(예 : '카페', '음식점')
    8. 시간 : 시간을 분류할 수 있는 키워드(예 : '퇴근 후', '새벽 감성')

    키워드 추출 조건:
    1. 메타데이터에 대해 카테고리별로 장소의 특징을 잘 설명하는 키워드를 추출하세요.
    2. 형용사 키워드는 명사형 어미로 변경해서 추출하세요 (예: 친절한 → 친절함, 조용한 → 조용함).
    3. 형용사와 명사를 활용하여 구체적인 키워드를 추출해주세요 (예. 맛있는 커피, 시원한 냉면)
    4. 동일하거나 유사한 의미의 표현은 하나로 통일해서 중복 없이 추출하세요.
    5. [음식점, 카페, 편의점, 영화관] 중에서만 음식/제품에 적합한 장소 카테고리 키워드를 선택하세요.
    5. 긍정적인 의미의 키워드만 추출하세요.
    6. 마지막으로, 장소의 분위기와 장점을 잘 살려 사장님이 직접 작성한 느낌의 소개글을 만들어주세요.  
    문장은 2~3줄, 따뜻한 말투로 작성하며 이모티콘은 장소에 맞게 1개만 사용해주세요.

    출력 형식:
    장소 소개글:

    장소 키워드:
    {{
    "음식/제품": [],
    "분위기/공간": [],
    "서비스/직원": [],
    "가격/가성비": [],
    "접근성/편의시설": [],
    "방문 목적": [],
    "장소 카테고리": [],
    "시간": []
    }}
    """
    return prompt


def get_model(api_key: str, model_name: str) -> GenerativeModel:
    global _model_instance
    if _model_instance is None:
        genai.configure(api_key=api_key)
        _model_instance = genai.GenerativeModel(model_name)
    return _model_instance


def parse_output(llm_response: str):
    # 소개글 추출
    description_match = re.search(r"장소 소개글:\s*(.*?)\s*장소 키워드:", llm_response, re.DOTALL)
    description = description_match.group(1).strip() if description_match else ""

    # 키워드 딕셔너리 추출
    keywords_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
    if not keywords_match:
        keywords_match = re.search(r"장소 키워드:\s*(\{.*\})", llm_response, re.DOTALL)

    keywords_json = keywords_match.group(1).strip() if keywords_match else "{}"
    keywords = json.loads(keywords_json)

    return description, keywords


def post_processing(place_table, place_menu_table, place_facilities, place_reviews):
    prompt = get_prompt()
    metadata_block = {
        "place_name": place_table['name'].values,
        "menu": place_menu_table[['menu_name', 'price']].values,
        "facilities": place_facilities,
        "reviews": place_reviews['text'].values
    }

    final_prompt = prompt.format(metadata_block=metadata_block)

    model = get_model(settings.GOOGLE_API_KEY, settings.MODEL_NAME)

    response = model.generate_content(final_prompt)

    description, keywords = parse_output(response.text)

    if not place_table.empty:
        place_table.at[0, "description"] = description
        if "장소 카테고리" in keywords and keywords["장소 카테고리"]:
            place_table.at[0, "category"] = keywords["장소 카테고리"][0]

    if not place_menu_table.empty:
        keywords["음식/제품"].extend(place_menu_table['menu_name'].values)

    return place_table, keywords