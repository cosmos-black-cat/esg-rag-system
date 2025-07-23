# =============================================================================
# ESG關鍵字配置文件 - 簡化版本
# =============================================================================

# 簡化後只保留四個關鍵字的配置
ESG_KEYWORDS_CONFIG = {
    "指標1_再生塑膠材料": {
        "description": "再生塑膠相關材料使用情況",
        "type": "percentage_or_number",  # 期望的數據類型：數值或百分比
        "keywords": [
            "再生塑膠",
            "再生塑料", 
            "再生料",
            "再生pp"
        ]
    }
}

# 提取提示詞模板
EXTRACTION_PROMPTS = {
    "percentage_or_number": """
從以下文本中提取與關鍵字"{keyword}"相關的數值或百分比數據。

任務要求：
1. 尋找具體的數字、百分比、重量、金額等數值資訊
2. 包括單位信息（如kg、噸、%、元等）
3. 如果有多個相關數值，請選擇最重要的一個
4. 必須包含數值或百分比才算找到，純文字描述不算
5. 如果沒有找到具體數值或百分比，請回答"未提及"

文本內容：
{context}

請以JSON格式回答：
{{
    "found": true/false,
    "value": "數值含單位或百分比或未提及",
    "confidence": 0-1之間的信心分數,
    "explanation": "簡短說明數值的含義和來源"
}}
"""
}

# =============================================================================
# 輔助函數
# =============================================================================

def get_all_keywords():
    """獲取所有關鍵字的列表"""
    all_keywords = []
    for indicator, config in ESG_KEYWORDS_CONFIG.items():
        all_keywords.extend(config["keywords"])
    return all_keywords

def get_keywords_by_indicator(indicator_name):
    """根據指標名稱獲取關鍵字"""
    return ESG_KEYWORDS_CONFIG.get(indicator_name, {}).get("keywords", [])

def get_indicator_by_keyword(keyword):
    """根據關鍵字找到對應的指標"""
    for indicator, config in ESG_KEYWORDS_CONFIG.items():
        if keyword in config["keywords"]:
            return indicator
    return None

def get_data_type_by_keyword(keyword):
    """根據關鍵字獲取期望的數據類型"""
    indicator = get_indicator_by_keyword(keyword)
    if indicator:
        return ESG_KEYWORDS_CONFIG[indicator]["type"]
    return "percentage_or_number"  # 默認為數值或百分比類型

def print_keywords_summary():
    """打印關鍵字配置摘要"""
    print("📋 簡化後的ESG關鍵字配置:")
    print("=" * 50)
    
    total_keywords = 0
    for indicator, config in ESG_KEYWORDS_CONFIG.items():
        keywords = config["keywords"]
        total_keywords += len(keywords)
        print(f"\n📊 {indicator}")
        print(f"   類型: {config['type']}")
        print(f"   關鍵字數量: {len(keywords)}")
        print(f"   關鍵字: {', '.join(keywords)}")
    
    print(f"\n總計: {total_keywords} 個關鍵字")
    print("=" * 50)

if __name__ == "__main__":
    print_keywords_summary()