"""
게시글 생성 서비스 모듈

이 모듈은 게시글 생성 서비스의 핵심 비즈니스 로직을 구현합니다.
LangChain을 사용하여 be 서버 입력을 활용해서 장소 추천 게시글을 생성합니다.

주요 구성요소:
    - GeneratorService: 추천 서비스 클래스
"""

import json

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas.moment_schema import GenerateRequest, GenerateResponse

class GeneratorService:
    """
    게시글 생성 서비스 모듈

    이 모듈은 게시글 생성 서비스의 핵심 비즈니스 로직을 구현합니다.
    LangChain을 사용하여 BE 서버 입력을 활용해서 장소 추천 게시글을 생성합니다.
    
    Attributes:
        llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
        chain: 키워드 추출을 위한 LangChain 체인
    """
    
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
    ):
        """
        RecommenderService 초기화
        
        Args:
            llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
        """
        self.llm = llm
        self.chain = self._create_chain()
    
    def _create_chain(self):
        """
        게시글 작성을 위한 LangChain 체인 생성
        
        Returns:
            RunnablePassthrough: 생성된 LangChain 체인
        """
        prompt = ChatPromptTemplate.from_template("""
            당신은 장소 게시글 작성 AI 돌핀입니다.

            아래에 제시된 장소의 정보를 참고해서 해당 장소를 방문한 경험을 작성하세요.

            장소 정보:
            {place_info}

            게시글 생성 조건:
            당신의 이름은 '돌핀'으로, 재밌는 방문 경험을 통해 장소를 추천하는 AI입니다.
            "content"는 게시글의 내용으로. 하루에 한 게시글만 올라갑니다. 그렇기에 사용자들이 매일 기대할 수 있도록 재치있고 재미있는 이야기를 작성하세요.
            게시글은 돌핀의 일상 썰로 이루어져 있으므로 장소 정보를 활용하여 하나의 상황을 컨셉으로 일관되게 작성하세요. Ex) 퇴근 후 힐링, 간단하고 빠른 점심, 애인과 다툼 후 화해, 더운 날씨
            주 사용자는 20대 청년으로 예시와 같은 인터넷 밈, 말투를 활용해서 게시글을 작성하세요.
            6~7줄 분량, 내용으로 작성하세요.
            "title"은 게시글의 제목으로, 게시글 내용을 재치있게 표현하는 제목으로 작성하세요.

            주의 사항:
            이모티콘, 해시테그를 사용하지 마세요.
            장소 정보를 모두 활용하지 않아도 됩니다.
            게시글 중간에 추천을 위한 억지스러운 문장이 들어가면 안됩니다.
            게시글 컨셉에 맞는 장소 정보만 활용하며, 광고와 같이 느껴지면 절대 안됩니다.
            어그로를 끌 수 있게 작성히되, 욕설, 비속어가 포함되면 절대 안됩니다.

            예시:
            {{
                "title": "돼지고기 좋아하면 개추 ㄱㄱ",
                "content": "라고 할 줄 알았냐 소고기가 더 비싼데 소고기가 맛있겠지"
            }}

            출력 형식:
            {{
                "title": "게시글 제목",
                "content": "게시글 내용"
            }}
            """)
        
        return prompt | self.llm
    
    async def generate_moment(self, place_info: GenerateRequest) -> GenerateResponse:
        """
        장소 정보를 활용해서 게시글을 생성합니다.
                
        Args:
            place_info (GenerateRequest): 장소 정보
            
        Returns:
            GenerateResponse: 생성된 게시글
            
        Raises:
            Exception: 게시글 생성 과정에서 오류가 발생한 경우
        """
        try:
            response = await self.chain.ainvoke({"place_info": place_info})
            moment_str = response.content
            
            # JSON 문자열에서 게시글 딕셔너리 추출
            start_idx = moment_str.find("{")
            end_idx = moment_str.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("게시글 생성 실패: JSON 형식이 올바르지 않습니다.")
            
            moment_json = moment_str[start_idx:end_idx]
            moment = json.loads(moment_json)

            moment["place_id"] = place_info.id

            return moment
            
        except Exception as e:
            raise Exception(f"게시글 생성 중 오류 발생: {str(e)}")