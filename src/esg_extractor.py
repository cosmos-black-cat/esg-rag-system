import json
import os
import re
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(str(Path(__file__).parent))
from config import *

# 簡化後的關鍵字配置
SIMPLIFIED_KEYWORDS = ["再生塑膠", "再生塑料", "再生料", "再生pp"]

@dataclass
class FilteredResult:
    """兩段式篩選結果"""
    keyword: str
    value: str  # 數值或百分比
    paragraph: str  # 完整段落
    page_number: str  # 頁碼
    confidence: float
    source_file: str  # 來源檔案名稱

class TwoStageExtractor:
    """兩段式ESG資料篩選提取器"""
    
    def __init__(self, vector_db_path: str = None):
        """初始化提取器"""
        
        if vector_db_path is None:
            vector_db_path = VECTOR_DB_PATH
            
        self.vector_db_path = vector_db_path
        self.keywords = SIMPLIFIED_KEYWORDS
        
        # 初始化embedding模型
        print("📱 載入embedding模型...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # 載入向量資料庫
        print("📚 載入向量資料庫...")
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"向量資料庫不存在: {vector_db_path}")
        
        self.vector_db = FAISS.load_local(
            vector_db_path, 
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # 初始化LLM
        print("🤖 初始化Gemini LLM...")
        if not GOOGLE_API_KEY:
            raise ValueError("請設置 GOOGLE_API_KEY 環境變數")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )

    def stage1_filter_documents(self) -> Dict[str, List[Document]]:
        """第一階段：篩選包含關鍵字的文檔"""
        print("\n🔍 第一階段：篩選包含關鍵字的ESG報告書...")
        
        # 獲取所有文檔
        all_docs = []
        for i in range(self.vector_db.index.ntotal):
            # 這裡需要根據實際的向量資料庫結構來調整
            # 目前先用搜尋的方式來獲取文檔
            pass
        
        # 為每個關鍵字搜尋相關文檔
        filtered_docs = {}
        
        for keyword in self.keywords:
            print(f"  🔎 搜尋關鍵字: {keyword}")
            
            # 搜尋包含該關鍵字的文檔
            search_results = self.vector_db.similarity_search_with_score(
                keyword, 
                k=50  # 增加搜尋數量以獲得更多相關文檔
            )
            
            # 篩選確實包含關鍵字的文檔
            relevant_docs = []
            for doc, score in search_results:
                if keyword in doc.page_content:
                    relevant_docs.append(doc)
            
            filtered_docs[keyword] = relevant_docs
            print(f"    ✓ 找到 {len(relevant_docs)} 個相關文檔")
        
        return filtered_docs

    def extract_numbers_and_percentages(self, text: str) -> List[str]:
        """從文本中提取數值和百分比"""
        patterns = [
            r'\d+\.?\d*\s*%',  # 百分比：25.5%, 30%
            r'\d+\.?\d*\s*[kKgG]+',  # 重量：100KG, 50kg
            r'\d+\.?\d*\s*噸',  # 噸數：1000噸
            r'\d+\.?\d*\s*萬噸',  # 萬噸：5.5萬噸
            r'\d+\.?\d*\s*公斤',  # 公斤
            r'\d+\.?\d*\s*倍',  # 倍數：3倍
            r'\d+\.?\d*\s*次',  # 次數：10次
            r'\d+\.?\d*\s*年',  # 年份相關：5年
            r'\d{1,3}(?:,\d{3})*\.?\d*',  # 大數字：1,000, 50,000
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        
        return list(set(numbers))  # 去重

    def stage2_extract_numeric_data(self, filtered_docs: Dict[str, List[Document]]) -> List[FilteredResult]:
        """第二階段：提取包含數值或百分比的段落"""
        print("\n📊 第二階段：提取包含數值或百分比的段落...")
        
        results = []
        
        for keyword, docs in filtered_docs.items():
            print(f"\n  📋 處理關鍵字: {keyword} ({len(docs)} 個文檔)")
            
            for doc in docs:
                # 檢查段落是否包含數值或百分比
                numbers = self.extract_numbers_and_percentages(doc.page_content)
                
                if numbers:  # 如果找到數值或百分比
                    # 使用LLM來確認這些數值是否與關鍵字相關
                    relevant_value = self.verify_numeric_relevance(keyword, doc.page_content, numbers)
                    
                    if relevant_value:
                        result = FilteredResult(
                            keyword=keyword,
                            value=relevant_value,
                            paragraph=doc.page_content.strip(),
                            page_number=f"第{doc.metadata.get('page', 'unknown')}頁",
                            confidence=0.8,  # 這裡可以根據LLM的回應調整
                            source_file=doc.metadata.get('source', 'unknown')
                        )
                        results.append(result)
                        print(f"    ✓ 找到相關數值: {relevant_value}")
        
        return results

    def verify_numeric_relevance(self, keyword: str, text: str, numbers: List[str]) -> Optional[str]:
        """使用LLM驗證數值是否與關鍵字相關"""
        
        prompt = f"""
請分析以下文本中的數值是否與關鍵字"{keyword}"直接相關。

文本內容：
{text}

找到的數值：{', '.join(numbers)}

請判斷：
1. 這些數值中哪一個與"{keyword}"最相關？
2. 該數值代表什麼含義？

如果沒有直接相關的數值，請回答"無相關數值"。
如果有相關數值，請只回答最相關的那一個數值（包含單位）。

回答格式：數值 或 無相關數值
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            
            # 如果回答不是"無相關數值"，則返回該數值
            if result != "無相關數值" and result not in ["沒有", "無", "未找到"]:
                return result
            else:
                return None
        except Exception as e:
            print(f"    ⚠️ LLM驗證失敗: {e}")
            # 如果LLM失敗，返回第一個找到的數值
            return numbers[0] if numbers else None

    def run_two_stage_extraction(self) -> List[FilteredResult]:
        """執行完整的兩段式提取"""
        print("🚀 開始兩段式ESG資料提取...")
        print("=" * 60)
        
        # 第一階段：篩選文檔
        filtered_docs = self.stage1_filter_documents()
        
        # 統計第一階段結果
        total_docs = sum(len(docs) for docs in filtered_docs.values())
        print(f"\n📈 第一階段統計:")
        print(f"  篩選出文檔總數: {total_docs}")
        for keyword, docs in filtered_docs.items():
            print(f"  {keyword}: {len(docs)} 個文檔")
        
        # 第二階段：提取數值資料
        results = self.stage2_extract_numeric_data(filtered_docs)
        
        print(f"\n📊 第二階段統計:")
        print(f"  找到包含數值的段落: {len(results)} 個")
        
        return results

    def save_results_to_csv(self, results: List[FilteredResult]) -> str:
        """將結果保存為CSV檔案"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = RESULTS_PATH / f"two_stage_extraction_{timestamp}.csv"
        
        # 轉換為DataFrame
        data = []
        for result in results:
            data.append({
                '關鍵字': result.keyword,
                '數值或比例': result.value,
                '段落內容': result.paragraph,
                '頁碼': result.page_number,
                '信心分數': result.confidence,
                '來源檔案': result.source_file
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 結果已保存至: {csv_path}")
        return str(csv_path)

    def save_results_to_excel(self, results: List[FilteredResult]) -> str:
        """將結果保存為Excel檔案，包含統計資訊"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = RESULTS_PATH / f"two_stage_extraction_{timestamp}.xlsx"
        
        # 轉換為DataFrame
        data = []
        for result in results:
            data.append({
                '關鍵字': result.keyword,
                '數值或比例': result.value,
                '段落內容': result.paragraph,
                '頁碼': result.page_number,
                '信心分數': result.confidence,
                '來源檔案': result.source_file
            })
        
        df = pd.DataFrame(data)
        
        # 創建統計資訊
        stats_data = []
        for keyword in self.keywords:
            keyword_results = [r for r in results if r.keyword == keyword]
            stats_data.append({
                '關鍵字': keyword,
                '找到數量': len(keyword_results),
                '平均信心分數': np.mean([r.confidence for r in keyword_results]) if keyword_results else 0,
                '不重複數值': len(set([r.value for r in keyword_results]))
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # 寫入Excel文件
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='詳細結果', index=False)
            stats_df.to_excel(writer, sheet_name='統計摘要', index=False)
        
        print(f"\n📊 Excel報告已保存至: {excel_path}")
        return str(excel_path)

def main():
    """主函數"""
    try:
        # 初始化提取器
        extractor = TwoStageExtractor()
        
        # 執行兩段式提取
        results = extractor.run_two_stage_extraction()
        
        if results:
            # 保存結果
            csv_path = extractor.save_results_to_csv(results)
            excel_path = extractor.save_results_to_excel(results)
            
            # 顯示結果摘要
            print("\n" + "=" * 60)
            print("🎉 兩段式提取完成！")
            print("=" * 60)
            print(f"📋 總共找到: {len(results)} 個包含數值的相關段落")
            
            # 按關鍵字統計
            for keyword in extractor.keywords:
                count = len([r for r in results if r.keyword == keyword])
                print(f"  📊 {keyword}: {count} 個結果")
            
            print(f"\n📁 檔案已保存:")
            print(f"  CSV: {csv_path}")
            print(f"  Excel: {excel_path}")
            
        else:
            print("❌ 未找到任何符合條件的結果")
            
    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()