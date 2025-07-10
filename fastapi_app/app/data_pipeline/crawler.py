import re
import time
import requests
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from app.core.config import settings

def crawl_place_table(place_id: int, soup: BeautifulSoup, KAKAO_API_KEY: str) -> pd.DataFrame:
    """
    DB place_table 장소 메타 데이터 크롤링
    """
    # --------------------- 이름 ---------------------
    name_tag = soup.select_one("h2.tit_location")
    name = name_tag.get_text(strip=True) if name_tag else None

    if not name:
        meta_title = soup.select_one('meta[property="og:title"]')
        name = meta_title["content"].strip() if meta_title else None

    # --------------------- 전화번호 ---------------------
    phone = None
    for unit in soup.select("div.unit_default"):
        if unit.select_one("h5.tit_info span.ico_call2"):  # 전화 아이콘이 있는 섹션만 찾기
            phone_tag = unit.select_one("div.detail_info span.txt_detail")
            if phone_tag:
                temp = phone_tag.get_text(strip=True)
                if temp and "연락처" not in temp:
                    phone = temp
            break  # 전화번호는 하나만 존재하므로 찾으면 종료

    # --------------------- 도로명주소 ---------------------
    road_address = None
    for unit in soup.select("div.unit_default"):
        title_span = unit.select_one("h5.tit_info span.ico_address")
        if title_span:
            addr_tag = unit.select_one("div.detail_info span.txt_detail")
            road_address = addr_tag.get_text(strip=True) if addr_tag else None
            break

    # --------------------- 좌표 (도로명 기준) ---------------------
    address_query = road_address or name
    lon, lat = None, None
    if address_query:
        address_query = re.sub(r"\(우\)?\d{5}", "", address_query).strip()
        try:
            resp = requests.get(
                "https://dapi.kakao.com/v2/local/search/address.json",
                headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"},
                params={"query": address_query},
                timeout=5
            )
            result = resp.json().get("documents")
            if result:
                lon = float(result[0]["x"])
                lat = float(result[0]["y"])
        except Exception as e:
            print(f"[WARNING] 좌표 변환 실패: {address_query} → {e}")

    location = f"SRID=4326;POINT({lon} {lat})" if lon and lat else ""

    # --------------------- 이미지 ---------------------
    temp_image_path = settings.TEMP_IMAGE_PATH
    s3_image_path = settings.S3_IMAGE_PATH
    try:
        first_img = soup.select_one("div.board_photo img")
        if first_img and first_img.get("src"):
            src = first_img["src"]
            if src.startswith("//"):
                src = "https:" + src

            r = requests.get(src, stream=True)
            r.raise_for_status()
            with open(f"{temp_image_path}/{place_id}.jpg", "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)

            image_url = f"{s3_image_path}/{place_id}.jpg"

    except Exception as e:
        print(f"[WARN] 이미지 저장 실패: {e}")
        image_url = None
        
    # --------------------- 결과 리턴 ---------------------
    place_table = {
        "id": place_id,
        "place_category": None,
        "location": location,
        "description": None,
        "image_url": image_url,
        "name": name,
        "phone": phone,
        "lot_address": None,
        "road_address": road_address,
        "created_at": None,
        "updated_at": None
    }

    return pd.DataFrame([place_table])


def crawl_place_hours_table(place_id: int, soup: BeautifulSoup) -> pd.DataFrame:
    """
    DB place_hour_table 장소 영업시간 데이터 크롤링
    """
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    valid_days = set(weekdays + ["매일", "브레이크타임"])
    all_rows: list[dict] = []

    def _extract_opening_info(section: BeautifulSoup) -> list[dict]:
        out = []
        for row in section.select("div.line_fold"):
            day_tag = row.find("span", class_="tit_fold")
            datas   = [s.get_text(" ", strip=True) for s in row.select("span.txt_detail")]
            if not datas:
                continue

            if day_tag:
                day = day_tag.get_text(strip=True)
                out.append({"day": day, "details": datas})
            else:
                for raw in datas:
                    parts = raw.replace("\xa0", " ").split()
                    if len(parts) >= 2:
                        out.append({"day": parts[0], "details": [" ".join(parts[1:])]})
        return out

    def _normalize_opening_info(pid: int, items: list[dict]) -> list[dict]:
        rows = []
        for item in items:
            # 요일 파싱 Ex) '일(7/6)' → '일'
            day_raw = re.sub(r"\(.*?\)", "", item["day"]).strip()

            # 휴무·라스트오더·잘못된 요일은 제외
            if any(x in day_raw for x in ["휴무", "라스트오더"]) or day_raw not in valid_days:
                continue

            days = weekdays if day_raw in ("매일", "브레이크타임") else [day_raw]

            for detail in item["details"]:
                if any(x in detail for x in ["휴무", "라스트오더"]):
                    continue

                is_break  = "브레이크타임" in detail or day_raw == "브레이크타임"
                time_part = detail.replace("브레이크타임", "").strip()

                if "~" in time_part:
                    t = [s.strip() for s in time_part.split("~")]
                    if len(t) != 2 or not all(":" in x for x in t):
                        continue
                    open_t, close_t = t
                else:
                    open_t = close_t = time_part

                for d in days:
                    rows.append(
                        dict(
                            place_id=pid,
                            day_of_week=d,
                            open_time=open_t,
                            close_time=close_t,
                            is_break_time=bool(is_break),
                        )
                    )
        return rows

    section = soup.find("div", id="foldDetail2")
    if not section:
        return pd.DataFrame(columns=["place_id", "day_of_week", "open_time", "close_time", "is_break_time"])

    all_rows.extend(_normalize_opening_info(place_id, _extract_opening_info(section)))

    place_hours_table = pd.DataFrame(
        all_rows,
        columns=["place_id", "day_of_week", "open_time", "close_time", "is_break_time"]
    )

    return place_hours_table


def crawl_place_facilities(soup: BeautifulSoup) -> list[str]:
    """
    키워드 추출에 활용할 장소 시설 정보 데이터 크롤링
    """
    place_facilities = []
    # 시설 정보 추출
    facility_section = soup.select_one("div.wrap_storeinfo.wrap_facilities")
    if facility_section:
        for span in facility_section.select("span.txt_svc"):
            text = span.get_text(strip=True)
            if text:
                place_facilities.append(text)

    # 해시태그 정보 추출
    hashtag_section = soup.select("a.link_detail")
    for tag in hashtag_section:
        text = tag.get_text(strip=True)
        if text.startswith("#"):
            place_facilities.append(text.lstrip("#"))  # '#' 제거

    return place_facilities


def crawl_place_menu_table(place_id: int, soup: BeautifulSoup) -> pd.DataFrame:
    """
    DB place_menu_table 장소 메뉴 데이터 크롤링
    """
    place_menu_table = []

    menu_blocks = soup.find_all('div', class_='info_goods')
    for block in menu_blocks:
        menu_name_tag = block.find('strong', class_='tit_item')
        price_tag = block.find('p', class_='desc_item')

        menu_name = menu_name_tag.text.strip() if menu_name_tag else None
        price = price_tag.text.strip() if price_tag else None

        # 가격이 '-' 또는 공백일 경우 None 처리
        if price == '-' or not price:
            price = None

        if menu_name:  # 이름이 있는 항목만 유효
            place_menu_table.append({
                "place_id": place_id,
                "menu_name": menu_name,
                "price": price
            })

    return pd.DataFrame(place_menu_table, columns=["place_id", "menu_name", "price"])


def crawl_place_reviews(soup: BeautifulSoup) -> pd.DataFrame:
    """
    키워드 추출에 활용할 장소 리뷰 데이터 크롤링
    """
    place_reviews = []
    review_items = soup.select('ul.list_review > li')

    for r in review_items:
        try:
            # 별점
            score_tag = r.select_one('div.info_grade span.screen_out:nth-of-type(2)')
            score = score_tag.text.strip() if score_tag else ""

            # 리뷰 본문
            text_tag = r.select_one('p.desc_review')
            text = text_tag.text.strip() if text_tag else ""

            if not text:  # 본문이 없으면 저장 안 함
                continue
            
            place_reviews.append({
                'score': float(score),
                'text': text
            })

        except Exception as e:
            print(f"❗ 리뷰 파싱 오류: {e}")
            continue
    df = pd.DataFrame(place_reviews)
    return df.sort_values(by='score', ascending=False).reset_index(drop=True)


def crawling(place_id):
    """
    전체 데이터 페이지 단위 크롤링
    """
    # Selenium 드라이버 설정
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = f'https://place.map.kakao.com/{place_id}'
    driver.get(url)
    
    # --------------메인 페이지 크롤링--------------
    try:
        # JS 로딩이 완료될 때까지 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h3.tit_place"))
        )
        time.sleep(1)
    except:
        print(f"[ERROR] 메인 페이지 로딩 실패: {url}")
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    place_table = crawl_place_table(place_id, soup, settings.KAKAO_API_KEY)
    place_hours_table = crawl_place_hours_table(place_id, soup)
    place_facilities = crawl_place_facilities(soup)

    # --------------메뉴 페이지 크롤링--------------
    try:
        info_tab = driver.find_element(By.XPATH, "//a[@href='#menuInfo']")
        driver.execute_script("arguments[0].click();", info_tab)
        time.sleep(1)
    except:
        print(f"[ERROR] 메뉴 페이지 로딩 실패: {url}")
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    place_menu_table = crawl_place_menu_table(place_id, soup)

    # --------------후기 페이지 크롤링--------------
    try:
        info_tab = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//a[@href='#comment']"))
        )
        driver.execute_script("arguments[0].click();", info_tab)
        time.sleep(1)

        # 무한 스크롤
        scroll_pause_time = 0.5
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        # 더보기 버튼 모두 클릭 (여러 번 탐색)
        for _ in range(3):  # 최대 3회 반복
            try:
                buttons = driver.find_elements(By.CSS_SELECTOR, "span.btn_more")
                if not buttons:
                    break
                for btn in buttons:
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.1)
                    except:
                        continue
            except Exception as e:
                print(f"❗ 더보기 클릭 실패: {e}")
                break
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        place_reviews = crawl_place_reviews(soup)
        
    except Exception as e:
        print(f"[ERROR] 후기 페이지 로딩 실패: {url}")
        place_reviews = pd.DataFrame(columns=["score", "text"])


    return place_table, place_hours_table, place_facilities, place_menu_table, place_reviews