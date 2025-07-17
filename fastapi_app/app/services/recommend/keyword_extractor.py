import json
import re

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class KeywordExtractor:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.chain = self._create_chain()

    def _create_chain(self):
        """
        키워드 추출을 위한 LangChain 체인 생성
        
        Returns:
            RunnablePassthrough: 생성된 LangChain 체인
        """
        prompt = ChatPromptTemplate.from_template("""
            당신은 장소 검색 키워드 추출 AI입니다.

            아래에 제시된 사용자의 요청에서 사용자가 원하는 장소의 특징을 잘 나타내는 키워드를 추출하세요.

            사용자 요청:
            {user_input}

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
            1. 사용자 요청에 대해 카테고리별로 장소의 특징을 잘 설명하는 키워드를 추출하세요.
            2. 형용사 키워드는 명사형 어미로 변경해서 추출하세요 (예: 친절한 → 친절함, 조용한 → 조용함).
            3. 형용사와 명사를 활용하여 구체적인 키워드를 추출해주세요 (예. 맛있는 커피, 시원한 냉면)
            4. 동일하거나 유사한 의미의 표현은 하나로 통일해서 중복 없이 추출하세요.
            5. [음식점, 카페, 편의점, 영화관] 중에서만 음식/제품에 적합한 장소 카테고리 키워드를 선택

            출력 형식:
            {{
            "음식/제품": [],
            "분위기/공간": [],
            "서비스/직원": [],
            "가격/가성비": [],
            "접근성/편의시설": [],
            "방문 목적": [],
            "장소 카테고리" : [],
            "시간" : []
            }}
            """)
        
        return prompt | self.llm

    async def extract(self, user_input: str) -> dict:
        response = await self.chain.ainvoke({"user_input": user_input})
        return self._parse_response(response.content)

    def _parse_response(self, raw: str) -> dict:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("키워드 추출 실패: JSON 형식이 올바르지 않습니다.")
        parsed = json.loads(raw[start:end])

        categories = None
        keywords = None

        place_category = None

        for category, kw_list in parsed.items():
            if not kw_list:
                continue
            elif category == '장소 카테고리':
                place_category = kw_list[0]
                continue
            for keyword in kw_list:
                if categories == None and keywords == None:
                    categories, keywords = [], []
                categories.append(category)
                keywords.append(keyword)

        return parsed, categories, keywords, place_category

class ClovaXKeywordExtractor:
    """
    ClovaX LLM 기반 키워드 추출기
    """
    def __init__(self, llm):
        self.llm = llm
        self.chain = self._create_chain()

    def extract_final_json(text: str):
        # 모든 JSON 블록 (```json 포함 or { ... }) 추출
        json_blocks = re.findall(r"```json\s*({[\s\S]*?})\s*```", text)  # ```json 블록
        if not json_blocks:
            json_blocks = re.findall(r"({[\s\S]*?})", text)  # 중괄호 블록만

        if not json_blocks:
            return text.strip()
        return json_blocks[-1].strip()
    
    def _create_chain(self):
        prompt = ChatPromptTemplate.from_template("""
            당신은 장소 검색 키워드 추출 AI입니다.

            아래에 제시된 입력에서 사용자가 원하는 장소의 특징을 잘 나타내는 키워드를 추출하세요.

            입력:
            {user_input}

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
            1. 사용자 요청에 대해 카테고리별로 장소의 특징을 잘 설명하는 키워드를 추출하세요.
            2. 형용사 키워드는 명사형 어미로 변경해서 추출하세요 (예: 친절한 → 친절함, 조용한 → 조용함).
            3. 형용사와 명사를 활용하여 구체적인 키워드를 추출해주세요 (예. 맛있는 커피, 시원한 냉면)
            4. 동일하거나 유사한 의미의 표현은 하나로 통일해서 중복 없이 추출하세요.
            5. [음식점, 카페, 편의점, 영화관] 중에서만 음식/제품에 적합한 장소 카테고리 키워드를 선택

            출력은 밑의 정답 형식에대한 값만 출력하세요 이외의 내용은 출력하지 않습니다.
            정답:
            {{
            "음식/제품": [],
            "분위기/공간": [],
            "서비스/직원": [],
            "가격/가성비": [],
            "접근성/편의시설": [],
            "방문 목적": [],
            "장소 카테고리" : [],
            "시간" : []
            }}
            """)
        return prompt | self.llm | self.extract_final_json

    async def extract(self, user_input: str) -> dict:
        response = await self.chain.ainvoke({"user_input": user_input})
        return response

    def _parse_response(self, text: str) -> dict:
        # 기존 KeywordExtractor의 _parse_response 복사
        try:
            parsed = json.loads(text)
        except Exception:
            raise ValueError("키워드 추출 실패: JSON 형식이 올바르지 않습니다.")

        categories = None
        keywords = None

        place_category = None

        for category, kw_list in parsed.items():
            if not kw_list:
                continue
            elif category == '장소 카테고리':
                place_category = kw_list[0]
                continue
            for keyword in kw_list:
                if categories == None and keywords == None:
                    categories, keywords = [], []
                categories.append(category)
                keywords.append(keyword)

        return parsed, categories, keywords, place_category