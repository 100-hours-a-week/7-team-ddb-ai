CATEGORY_MAP = {
    "음식/제품": "food_product",
    "분위기/공간": "ambience_space",
    "서비스/직원": "service_staff",
    "가격/가성비": "price_value",
    "접근성/편의시설": "accessibility",
    "방문 목적": "visit_purpose"
}

REVERSE_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()} 