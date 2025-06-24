# =============================================================================
# ESG關鍵字配置文件
# =============================================================================

# ESG六大指標的關鍵字配置
ESG_KEYWORDS_CONFIG = {
    "指標1_再生材料使用比例": {
        "description": "使用再生材料製造之比例中位數與普及度企業數量",
        "type": "percentage",  # 期望的數據類型
        "keywords": [
            # === 中文關鍵字 ===
            "再生材料使用比例",
            "再生塑膠佔比", 
            "rPET含量",
            "PCR材料使用",
            "導入再生料",
            
            # === 英文關鍵字 ===
            "recycled content ratio%",
            "PCR plastic usage",
            "rPET content",
            "% recycled materials",
            "recycled plastic usage rate",
            "PCR material use",
            "post-consumer plastic share",
            "Recycled Content Claim"
        ]
    },
    
    "指標2_碳排放數據比較": {
        "description": "歷年再生料碳排放數據比較",
        "type": "number",
        "keywords": [
            # === 中文關鍵字 ===
            "再生材料碳排放",
            "碳足跡比較", 
            "再生料減碳量",
            "歷年GHG排放量",
            "減碳數據",
            "碳排年表",
            "GHG減量",
            
            # === 英文關鍵字 ===
            "recycled material carbon footprint",
            "CO2 savings over years",
            "annual GHG reduction", 
            "CO2e savings by PCR",
            "carbon footprint of recycled materials",
            "GHG reduction by recycled content"
        ]
    },
    
    "指標3_資源循環效益比較": {
        "description": "歷年原生料與再生材料資源循環效益比較",
        "type": "percentage",
        "keywords": [
            # === 中文關鍵字 ===
            "原生料與再生料比較",
            "資源節省",
            "資源使用效率",
            "再生材料節能與新料比較", 
            "回收資源效率",
            
            # === 英文關鍵字 ===
            "virgin vs recycled resource efficiency",
            "material savings",
            "resource efficiency", 
            "virgin vs recycled materials",
            "environmental savings comparison",
            "recycled vs virgin environmental impact"
        ]
    },
    
    "指標4_產品延長壽命": {
        "description": "產品延長壽命因子",
        "type": "boolean_or_number",
        "keywords": [
            # === 中文關鍵字 ===
            "產品壽命",
            "延長使用年限",
            "可維修設計",
            "模組化設計",
            "耐用性提升",
            "維修服務年限",
            "延長保固",
            
            # === 英文關鍵字 ===
            "product lifespan",
            "durability",
            "modular design",
            "repairability", 
            "upgradeable",
            "extension service life"
        ]
    },
    
    "指標5_單位經濟效益": {
        "description": "單位經濟效益",
        "type": "number",
        "keywords": [
            # === 中文關鍵字 ===
            "每公斤成本",
            "每單位碳排放",
            "每單位效益",
            "再生塑膠性價比",
            "循環效益與成本比",
            
            # === 英文關鍵字 ===
            "unit economic benefit",
            "cost per kg",
            "CO2 per unit",
            "ROI of recycling",
            "lifecycle cost benefit", 
            "economic efficiency of recycling",
            "lifecycle unit cost",
            "unit GHG",
            "recycled vs virgin unit value"
        ]
    },
    
    "指標6_循環再製次數": {
        "description": "循環再製次數",
        "type": "number",
        "keywords": [
            # === 中文關鍵字 ===
            "循環次數",
            "再利用次數", 
            "再生次數",
            "可再製次數",
            "再加工循環",
            "材料耐久",
            "回收次數",
            
            # === 英文關鍵字 ===
            "recycling cycles",
            "number of reuse times",
            "remanufacturing times",
            "recycled loops",
            "recycled material durability",
            "durability of recycled material",
            "material degradation"
        ]
    }
}

# =============================================================================
# 關鍵字語義增強配置
KEYWORD_SEMANTIC_ENHANCEMENT = {
    # 指標1: 再生材料使用比例
    "再生材料使用比例": {
        "synonyms": ["再生料佔比", "回收材料比例", "循環材料使用率"],
        "context_clues": ["再生", "回收", "循環利用", "PCR", "rPET"],
        "expected_range": "0-100%",
        "typical_units": ["%", "百分比", "比例"]
    },
    "recycled content ratio%": {
        "synonyms": ["recycled content percentage", "recycled material ratio"],
        "context_clues": ["recycled", "content", "ratio", "percentage"],
        "expected_range": "0-100%",
        "typical_units": ["%", "percent"]
    },
    
    # 指標2: 碳排放相關
    "再生材料碳排放": {
        "synonyms": ["再生料碳足跡", "回收材料碳排", "循環材料碳排放"],
        "context_clues": ["碳排放", "CO2", "溫室氣體", "碳足跡"],
        "expected_range": "正數",
        "typical_units": ["噸CO2", "kg CO2", "tCO2e"]
    },
    "CO2 savings over years": {
        "synonyms": ["carbon savings", "CO2 reduction", "carbon emission reduction"],
        "context_clues": ["CO2", "carbon", "savings", "reduction"],
        "expected_range": "正數",
        "typical_units": ["tons CO2", "kg CO2", "tCO2e"]
    },
    
    # 指標4: 產品壽命相關
    "產品壽命": {
        "synonyms": ["使用壽命", "產品生命週期", "耐用年限"],
        "context_clues": ["壽命", "年限", "使用期間", "耐用"],
        "expected_range": "正數",
        "typical_units": ["年", "月", "天"]
    },
    "耐用性提升": {
        "synonyms": ["耐久性改善", "品質提升", "壽命延長"],
        "context_clues": ["耐用", "提升", "改善", "增強", "延長"],
        "expected_values": ["是", "否", "百分比提升"],
        "improvement_keywords": ["提升", "改善", "增強", "延長", "加強"]
    },
    
    # 指標5: 經濟效益相關
    "每公斤成本": {
        "synonyms": ["單位成本", "公斤成本", "單價"],
        "context_clues": ["成本", "價格", "費用", "每公斤", "單位"],
        "expected_range": "正數",
        "typical_units": ["元/kg", "$/kg", "元", "$"]
    },
    "unit economic benefit": {
        "synonyms": ["unit cost", "cost per unit", "economic efficiency"],
        "context_clues": ["unit", "cost", "economic", "benefit"],
        "expected_range": "正數",
        "typical_units": ["$/unit", "cost per"]
    },
    
    # 指標6: 循環次數相關
    "循環次數": {
        "synonyms": ["回收次數", "再利用次數", "循環週期"],
        "context_clues": ["循環", "次數", "回收", "再利用"],
        "expected_range": "正整數",
        "typical_units": ["次", "回", "週期"]
    },
    "recycling cycles": {
        "synonyms": ["reuse cycles", "recycling times", "circular cycles"],
        "context_clues": ["recycling", "cycles", "reuse", "times"],
        "expected_range": "正整數",
        "typical_units": ["times", "cycles"]
    }
}

def get_keyword_enhancement(keyword: str) -> dict:
    """獲取關鍵字的語義增強信息"""
    return KEYWORD_SEMANTIC_ENHANCEMENT.get(keyword, {
        "synonyms": [],
        "context_clues": [],
        "expected_range": "不限",
        "typical_units": []
    })

def create_enhanced_prompt(keyword: str, context: str, data_type: str) -> str:
    """創建增強版的提取提示"""
    
    enhancement = get_keyword_enhancement(keyword)
    base_prompt = EXTRACTION_PROMPTS.get(data_type, EXTRACTION_PROMPTS["number"])
    
    # 增加語義增強信息
    enhancement_text = f"""
## 關鍵字增強信息：
- 目標關鍵字：{keyword}
- 同義詞：{', '.join(enhancement.get('synonyms', []))}
- 上下文線索：{', '.join(enhancement.get('context_clues', []))}
- 預期範圍：{enhancement.get('expected_range', '不限')}
- 常見單位：{', '.join(enhancement.get('typical_units', []))}

## 額外提示：
- 尋找包含同義詞或上下文線索的句子
- 優先考慮預期範圍內的數值
- 注意常見單位的數據
"""
    
    return base_prompt.format(keyword=keyword, context=context) + enhancement_text
# =============================================================================

EXTRACTION_PROMPTS = {
    "percentage": """
從以下文本中提取與關鍵字"{keyword}"相關的百分比數據。

任務要求：
1. 尋找具體的百分比數值（如25%、0.3、30.5%等）
2. 如果找到多個數值，請選擇最相關的一個
3. 如果沒有找到具體數值，請回答"未提及"
4. 如果數據不清楚或模糊，請回答"數據不明確"

文本內容：
{context}

請以JSON格式回答：
{{
    "found": true/false,
    "value": "數值或未提及",
    "confidence": 0-1之間的信心分數,
    "explanation": "簡短說明為什麼選擇這個數值"
}}
""",
    
    "boolean_or_number": """
從以下文本中提取與關鍵字"{keyword}"相關的信息。

任務要求：
1. 如果明確提到有改善/提升/增加，請回答"是"
2. 如果明確提到沒有改善或下降，請回答"否"  
3. 如果有具體的改善數值或百分比，請提取具體數值
4. 如果沒有相關信息，請回答"未提及"

文本內容：
{context}

請以JSON格式回答：
{{
    "found": true/false,
    "value": "是/否/未提及或具體數值",
    "confidence": 0-1之間的信心分數,
    "explanation": "簡短說明判斷依據"
}}
""",
    
    "number": """
從以下文本中提取與關鍵字"{keyword}"相關的數值數據。

任務要求：
1. 尋找具體的數字、金額、數量、重量等數值
2. 包括單位信息（如kg、噸、元、次等）
3. 如果有多個相關數值，請選擇最重要的一個
4. 如果沒有找到相關數值，請回答"未提及"

文本內容：
{context}

請以JSON格式回答：
{{
    "found": true/false,
    "value": "數值含單位或未提及",
    "confidence": 0-1之間的信心分數,
    "explanation": "簡短說明數值的含義"
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
    return "number"  # 默認為數值類型

def print_keywords_summary():
    """打印關鍵字配置摘要"""
    print("📋 ESG關鍵字配置摘要:")
    print("=" * 50)
    
    total_keywords = 0
    for indicator, config in ESG_KEYWORDS_CONFIG.items():
        keyword_count = len(config["keywords"])
        total_keywords += keyword_count
        print(f"{indicator}:")
        print(f"  類型: {config['type']}")
        print(f"  關鍵字數量: {keyword_count}")
        print(f"  描述: {config['description']}")
        print()
    
    print(f"總計: {len(ESG_KEYWORDS_CONFIG)} 個指標, {total_keywords} 個關鍵字")

# =============================================================================
# 驗證配置
# =============================================================================

def validate_keywords_config():
    """驗證關鍵字配置的完整性"""
    errors = []
    
    # 檢查每個指標是否有必要的字段
    for indicator, config in ESG_KEYWORDS_CONFIG.items():
        if "type" not in config:
            errors.append(f"{indicator}: 缺少 'type' 字段")
        
        if "keywords" not in config:
            errors.append(f"{indicator}: 缺少 'keywords' 字段")
        elif not config["keywords"]:
            errors.append(f"{indicator}: 'keywords' 為空")
        
        if "description" not in config:
            errors.append(f"{indicator}: 缺少 'description' 字段")
    
    # 檢查是否有重複的關鍵字
    all_keywords = get_all_keywords()
    unique_keywords = set(all_keywords)
    if len(all_keywords) != len(unique_keywords):
        duplicates = [kw for kw in unique_keywords if all_keywords.count(kw) > 1]
        errors.append(f"重複的關鍵字: {duplicates}")
    
    # 檢查數據類型是否有效
    valid_types = {"percentage", "number", "boolean_or_number"}
    for indicator, config in ESG_KEYWORDS_CONFIG.items():
        if config.get("type") not in valid_types:
            errors.append(f"{indicator}: 無效的數據類型 '{config.get('type')}'")
    
    if errors:
        raise ValueError("關鍵字配置錯誤:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# 當模組被導入時自動驗證
if __name__ != "__main__":
    try:
        validate_keywords_config()
    except ValueError as e:
        print(f"⚠️  關鍵字配置警告: {e}")

# =============================================================================
# 主函數（用於測試）
# =============================================================================

if __name__ == "__main__":
    print("🧪 測試關鍵字配置...")
    try:
        validate_keywords_config()
        print("✅ 配置驗證通過！")
        print()
        print_keywords_summary()
        
        # 測試輔助函數
        print("\n🔍 測試輔助函數:")
        test_keyword = "再生材料使用比例"
        print(f"關鍵字 '{test_keyword}' 屬於指標: {get_indicator_by_keyword(test_keyword)}")
        print(f"期望數據類型: {get_data_type_by_keyword(test_keyword)}")
        
    except ValueError as e:
        print(f"❌ 配置錯誤: {e}")