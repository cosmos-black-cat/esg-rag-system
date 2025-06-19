# ESG關鍵字配置
ESG_KEYWORDS_CONFIG = {
    "指標1_再生材料使用比例": {
        "type": "percentage",
        "keywords": [
            # 中文關鍵字
            "再生材料使用比例", "再生塑膠佔比", "rPET含量", "PCR材料使用", "導入再生料",
            # 英文關鍵字
            "recycled content ratio%", "PCR plastic usage", "rPET content", 
            "% recycled materials", "recycled plastic usage rate", "PCR material use",
            "post-consumer plastic share", "Recycled Content Claim"
        ]
    },
    "指標2_碳排放數據比較": {
        "type": "number",
        "keywords": [
            # 中文關鍵字
            "再生材料碳排放", "碳足跡比較", "再生料減碳量", "歷年GHG排放量", 
            "減碳數據", "碳排年表", "GHG減量",
            # 英文關鍵字
            "recycled material carbon footprint", "CO2 savings over years", 
            "annual GHG reduction", "CO2e savings by PCR", 
            "carbon footprint of recycled materials", "GHG reduction by recycled content"
        ]
    },
    "指標3_資源循環效益比較": {
        "type": "percentage",
        "keywords": [
            # 中文關鍵字
            "原生料與再生料比較", "資源節省", "資源使用效率", "回收資源效率",
            "再生材料節能與新料比較",
            # 英文關鍵字
            "virgin vs recycled resource efficiency", "material savings", 
            "resource efficiency", "virgin vs recycled materials",
            "environmental savings comparison", "recycled vs virgin environmental impact"
        ]
    },
    "指標4_產品延長壽命": {
        "type": "boolean_or_number",
        "keywords": [
            # 中文關鍵字
            "產品壽命", "延長使用年限", "可維修設計", "模組化設計", 
            "耐用性提升", "維修服務年限", "延長保固",
            # 英文關鍵字
            "product lifespan", "durability", "modular design", "repairability", 
            "upgradeable", "extension service life"
        ]
    },
    "指標5_單位經濟效益": {
        "type": "number",
        "keywords": [
            # 中文關鍵字
            "每公斤成本", "每單位碳排放", "每單位效益", "再生塑膠性價比", 
            "循環效益與成本比",
            # 英文關鍵字
            "unit economic benefit", "cost per kg", "CO2 per unit", "ROI of recycling",
            "lifecycle cost benefit", "economic efficiency of recycling", 
            "lifecycle unit cost", "unit GHG", "recycled vs virgin unit value"
        ]
    },
    "指標6_循環再製次數": {
        "type": "number",
        "keywords": [
            # 中文關鍵字
            "循環次數", "再利用次數", "再生次數", "可再製次數", 
            "再加工循環", "材料耐久", "回收次數",
            # 英文關鍵字
            "recycling cycles", "number of reuse times", "remanufacturing times",
            "recycled loops", "recycled material durability", 
            "durability of recycled material", "material degradation"
        ]
    }
}

# 提取提示模板
EXTRACTION_PROMPTS = {
    "percentage": """
從以下文本中提取與關鍵字"{keyword}"相關的百分比數據。
如果找到具體數值，請提取百分比（如25%、0.3等）。
如果沒有找到具體數值，請回答"未提及"。

文本內容：
{context}

請以JSON格式回答：
{{
    "found": true/false,
    "value": "數值或未提及",
    "confidence": 0-1之間的信心分數,
    "explanation": "簡短說明"
}}
""",
    
    "boolean_or_number": """
從以下文本中提取與關鍵字"{keyword}"相關的信息。
如果提到有改善/提升/增加，請回答"是"。
如果提到沒有改善或下降，請回答"否"。
如果有具體的改善數值或百分比，請一併提取。
如果沒有相關信息，請回答"未提及"。

文本內容：
{context}

請以JSON格式回答：
{{
    "found": true/false,
    "value": "是/否/未提及或具體數值",
    "confidence": 0-1之間的信心分數,
    "explanation": "簡短說明"
}}
""",
    
    "number": """
從以下文本中提取與關鍵字"{keyword}"相關的數值數據。
請提取具體的數字、金額、數量等。
如果沒有找到相關數值，請回答"未提及"。

文本內容：
{context}

請以JSON格式回答：
{{
    "found": true/false,
    "value": "數值或未提及",
    "confidence": 0-1之間的信心分數,
    "explanation": "簡短說明"
}}
"""
}