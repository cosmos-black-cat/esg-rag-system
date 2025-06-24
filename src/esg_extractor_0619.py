import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
import numpy as np
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  langchain-google-genai 未安裝，將使用直接 API")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("❌ google-generativeai 未安裝")

sys.path.append(str(Path(__file__).parent))
from config import *
from keywords_config import ESG_KEYWORDS_CONFIG, EXTRACTION_PROMPTS, create_enhanced_prompt, get_keyword_enhancement

@dataclass
class ExtractionResult:
    """數據提取結果"""
    keyword: str
    indicator: str
    value: Optional[Union[float, str, bool]]
    value_type: str
    confidence: float
    source_text: str
    page_info: str

class ESGDataExtractor:
    def __init__(self, vector_db_path: str = None):
        """初始化ESG數據提取器"""
        
        if vector_db_path is None:
            vector_db_path = VECTOR_DB_PATH
            
        self.vector_db_path = vector_db_path
        self.keywords_config = ESG_KEYWORDS_CONFIG
        self.use_langchain = True  # 標記使用哪種API
        
        # 初始化embedding模型
        print("載入embedding模型...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # 載入向量資料庫
        print("載入向量資料庫...")
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"向量資料庫不存在: {vector_db_path}")
        
        self.db = FAISS.load_local(
            vector_db_path, 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # 初始化reranker
        print("載入reranker模型...")
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        
        # 初始化Gemini LLM
        print("初始化Gemini LLM...")
        
        if not GENAI_AVAILABLE:
            raise ImportError("請安裝 google-generativeai: pip install google-generativeai")
        
        try:
            # 方法1: 使用 LangChain 包裝器（如果可用）
            if LANGCHAIN_AVAILABLE:
                self.llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0,
                    convert_system_message_to_human=True
                )
                self.use_langchain = True
                print(f"✅ 使用 LangChain 包裝器，模型: {GEMINI_MODEL}")
            else:
                raise Exception("LangChain 不可用，使用直接 API")
                
        except Exception as e:
            print(f"LangChain 包裝器初始化失敗: {e}")
            print("改用直接 Google GenerativeAI API...")
            
            # 方法2: 直接使用 Google GenerativeAI
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # 清理模型名稱（移除 'models/' 前綴如果存在）
            model_name = GEMINI_MODEL
            if model_name.startswith('models/'):
                model_name = model_name[7:]
            
            self.llm = genai.GenerativeModel(model_name)
            self.use_langchain = False
            print(f"✅ 使用直接 API，模型: {model_name}")
        
        print("✅ ESG數據提取器初始化完成")

    def _dedup_documents(self, documents: List[Document]) -> List[Document]:
        """去除重複文檔"""
        seen_hashes = set()
        unique_docs = []
        for doc in documents:
            content_hash = hashlib.md5(doc.page_content.strip().encode("utf-8")).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        return unique_docs

    def search_and_rerank(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """搜尋並重新排序文檔"""
        # 初始搜尋
        results = self.db.similarity_search(query, k=k*2)
        
        # 去重
        unique_results = self._dedup_documents(results)
        
        if len(unique_results) == 0:
            return []
        
        # 使用BGE reranker重新排序
        pairs = [(query, doc.page_content) for doc in unique_results]
        inputs = self.reranker_tokenizer.batch_encode_plus(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits.view(-1)
        
        scores = scores.numpy()
        ranked = sorted(zip(unique_results, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def extract_data_with_llm(self, keyword: str, context: str, data_type: str) -> ExtractionResult:
        """使用Gemini LLM提取特定數據"""
        
        # 預處理文本：去除多餘空白和標點
        cleaned_context = self._clean_context(context)
        
        # 使用增強版提示
        enhanced_prompt = create_enhanced_prompt(keyword, cleaned_context, data_type)
        
        try:
            # 嘗試多次提取以提高準確性
            for attempt in range(2):  # 最多嘗試2次
                if self.use_langchain:
                    response = self.llm.invoke(enhanced_prompt)
                    content = response.content.strip()
                else:
                    response = self.llm.generate_content(enhanced_prompt)
                    content = response.text.strip()
                
                # 清理響應內容
                cleaned_content = self._clean_response(content)
                
                try:
                    result_json = json.loads(cleaned_content)
                    
                    # 驗證和後處理結果
                    validated_result = self._validate_and_process_result(
                        result_json, keyword, data_type, cleaned_context
                    )
                    
                    if validated_result["confidence"] > 0.3:  # 如果信心分數夠高就使用
                        return ExtractionResult(
                            keyword=keyword,
                            indicator="",
                            value=validated_result["value"],
                            value_type=data_type,
                            confidence=validated_result["confidence"],
                            source_text=validated_result.get("source_sentence", cleaned_context[:200] + "..."),
                            page_info=""
                        )
                except json.JSONDecodeError:
                    if attempt == 0:  # 第一次失敗，嘗試修復JSON
                        enhanced_prompt += "\n\n請確保回答是完全有效的JSON格式，不包含任何其他文字。"
                        continue
            
            # 如果所有嘗試都失敗，返回失敗結果
            return ExtractionResult(
                keyword=keyword,
                indicator="",
                value="提取失敗",
                value_type="error",
                confidence=0.0,
                source_text="JSON解析失敗",
                page_info=""
            )
            
        except Exception as e:
            print(f"Gemini提取失敗 ({keyword}): {e}")
            return ExtractionResult(
                keyword=keyword,
                indicator="",
                value="提取失敗",
                value_type="error",
                confidence=0.0,
                source_text=str(e),
                page_info=""
            )

    def _clean_context(self, context: str) -> str:
        """清理輸入文本"""
        import re
        
        # 移除多餘的空白字符
        context = re.sub(r'\s+', ' ', context)
        
        # 移除特殊字符和格式符號
        context = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()%/-]', '', context)
        
        # 確保長度適中（避免太長導致API問題）
        if len(context) > 3000:
            context = context[:3000] + "..."
        
        return context.strip()

    def _clean_response(self, content: str) -> str:
        """清理Gemini響應內容"""
        content = content.strip()
        
        # 移除markdown格式
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        if content.startswith('```'):
            content = content[3:]
        
        # 移除可能的前後說明文字，只保留JSON部分
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        return content.strip()

    def _validate_and_process_result(self, result_json: dict, keyword: str, 
                                   data_type: str, context: str) -> dict:
        """驗證和後處理提取結果"""
        
        # 基本字段檢查
        if not isinstance(result_json, dict):
            return {"value": "格式錯誤", "confidence": 0.0}
        
        value = result_json.get("value", "未提及")
        confidence = float(result_json.get("confidence", 0.0))
        
        # 根據數據類型進行特定驗證
        if data_type == "percentage":
            value, confidence = self._validate_percentage(value, confidence, keyword, context)
        elif data_type == "boolean_or_number":
            value, confidence = self._validate_boolean_or_number(value, confidence, keyword, context)
        elif data_type == "number":
            value, confidence = self._validate_number(value, confidence, keyword, context)
        
        # 確保信心分數在合理範圍內
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "value": value,
            "confidence": confidence,
            "source_sentence": result_json.get("source_sentence", ""),
            "reasoning": result_json.get("reasoning", "")
        }

    def _validate_percentage(self, value: str, confidence: float, keyword: str, context: str) -> tuple:
        """驗證百分比數據"""
        import re
        
        if value in ["未提及", "提取失敗", "數據不明確"]:
            return value, confidence
        
        # 檢查是否包含百分號或小數
        percentage_pattern = r'(\d+\.?\d*)%?'
        match = re.search(percentage_pattern, str(value))
        
        if match:
            num_str = match.group(1)
            try:
                num = float(num_str)
                
                # 驗證百分比範圍（0-100%，某些情況下可能超過100%）
                if 0 <= num <= 100:
                    if '%' not in str(value):
                        value = f"{num}%" if num > 1 else f"{num*100}%"
                    confidence = min(confidence + 0.1, 1.0)  # 格式正確，增加信心
                elif num > 100:
                    # 可能是累積增長或其他特殊情況
                    confidence = max(confidence - 0.2, 0.1)
                else:
                    confidence = max(confidence - 0.3, 0.1)
                    
            except ValueError:
                confidence = max(confidence - 0.4, 0.0)
        else:
            # 沒有找到數字格式
            confidence = max(confidence - 0.3, 0.0)
        
        # 檢查關鍵字相關性
        if keyword.lower() not in context.lower():
            confidence = max(confidence - 0.2, 0.0)
        
        return value, confidence

    def _validate_boolean_or_number(self, value: str, confidence: float, keyword: str, context: str) -> tuple:
        """驗證布爾或數值數據"""
        import re
        
        if value in ["未提及", "提取失敗"]:
            return value, confidence
        
        # 如果是是/否答案
        if value in ["是", "否"]:
            # 檢查文本中是否有支持證據
            positive_words = ["提升", "增加", "改善", "延長", "加強", "優化", "增強"]
            negative_words = ["下降", "減少", "惡化", "縮短", "降低", "退化"]
            
            context_lower = context.lower()
            has_positive = any(word in context for word in positive_words)
            has_negative = any(word in context for word in negative_words)
            
            if value == "是" and has_positive:
                confidence = min(confidence + 0.1, 1.0)
            elif value == "否" and has_negative:
                confidence = min(confidence + 0.1, 1.0)
            elif value == "是" and has_negative:
                confidence = max(confidence - 0.2, 0.1)
            elif value == "否" and has_positive:
                confidence = max(confidence - 0.2, 0.1)
        
        # 如果包含數值
        elif re.search(r'\d+', str(value)):
            # 增加對數值型回答的信心
            confidence = min(confidence + 0.05, 1.0)
        
        return value, confidence

    def _validate_number(self, value: str, confidence: float, keyword: str, context: str) -> tuple:
        """驗證數值數據"""
        import re
        
        if value in ["未提及", "提取失敗"]:
            return value, confidence
        
        # 檢查是否包含數字
        if re.search(r'\d+', str(value)):
            # 檢查是否有合理的單位
            units = ["元", "萬", "億", "kg", "噸", "次", "年", "月", "天", "%", "倍", "台", "件"]
            has_unit = any(unit in str(value) for unit in units)
            
            if has_unit:
                confidence = min(confidence + 0.1, 1.0)
            
            # 檢查數值是否合理（避免年份等無關數據）
            year_pattern = r'20\d{2}'
            if re.search(year_pattern, str(value)) and "年" not in keyword:
                confidence = max(confidence - 0.3, 0.1)
        else:
            confidence = max(confidence - 0.2, 0.1)
        
        return value, confidence

    def extract_all_keywords(self) -> Dict[str, List[ExtractionResult]]:
        """提取所有關鍵字的數據"""
        all_results = {}
        
        total_keywords = sum(len(config["keywords"]) for config in self.keywords_config.values())
        
        with tqdm(total=total_keywords, desc="提取ESG數據") as pbar:
            for indicator, config in self.keywords_config.items():
                print(f"\n📊 處理指標: {indicator}")
                indicator_results = []
                
                for keyword in config["keywords"]:
                    pbar.set_description(f"搜尋: {keyword}")
                    
                    # 搜尋相關文檔
                    search_results = self.search_and_rerank(keyword, k=3)
                    
                    if not search_results:
                        # 沒有找到相關文檔
                        result = ExtractionResult(
                            keyword=keyword,
                            indicator=indicator,
                            value="報告書中沒有提到",
                            value_type="not_found",
                            confidence=1.0,
                            source_text="",
                            page_info=""
                        )
                        indicator_results.append(result)
                        pbar.update(1)
                        continue
                    
                    # 合併相關文檔的內容
                    combined_context = ""
                    page_info = []
                    for doc, score in search_results:
                        combined_context += doc.page_content + "\n\n"
                        page_info.append(f"第{doc.metadata.get('page', 'unknown')}頁")
                    
                    # 使用Gemini提取數據
                    result = self.extract_data_with_llm(
                        keyword, 
                        combined_context, 
                        config["type"]
                    )
                    result.indicator = indicator
                    result.page_info = ", ".join(page_info)
                    
                    indicator_results.append(result)
                    pbar.update(1)
                
                all_results[indicator] = indicator_results
        
        return all_results

    def generate_summary_report(self, results: Dict[str, List[ExtractionResult]]) -> Dict:
        """生成摘要報告"""
        summary = {
            "total_keywords": 0,
            "found_keywords": 0,
            "not_found_keywords": 0,
            "high_confidence_results": 0,
            "indicators_summary": {}
        }
        
        for indicator, indicator_results in results.items():
            indicator_summary = {
                "total": len(indicator_results),
                "found": 0,
                "not_found": 0,
                "high_confidence": 0,
                "key_findings": []
            }
            
            for result in indicator_results:
                summary["total_keywords"] += 1
                
                if result.value_type == "not_found":
                    summary["not_found_keywords"] += 1
                    indicator_summary["not_found"] += 1
                else:
                    summary["found_keywords"] += 1
                    indicator_summary["found"] += 1
                    
                    if result.confidence > 0.7:
                        summary["high_confidence_results"] += 1
                        indicator_summary["high_confidence"] += 1
                    
                    # 收集重要發現
                    if result.confidence > 0.6 and result.value != "未提及":
                        indicator_summary["key_findings"].append({
                            "keyword": result.keyword,
                            "value": result.value,
                            "confidence": result.confidence
                        })
            
            summary["indicators_summary"][indicator] = indicator_summary
        
        return summary

    def save_results(self, results: Dict[str, List[ExtractionResult]], 
                    summary: Dict, output_path: str = None):
        """保存結果到JSON文件"""
        
        if output_path is None:
            output_path = os.path.join(RESULTS_PATH, "esg_extraction_results.json")
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 轉換ExtractionResult到可序列化的格式
        serializable_results = {}
        for indicator, indicator_results in results.items():
            serializable_results[indicator] = [
                asdict(r) for r in indicator_results
            ]
        
        output_data = {
            "extraction_results": serializable_results,
            "summary": summary,
            "metadata": {
                "total_indicators": len(self.keywords_config),
                "extraction_method": "RAG + BGE Reranker + Gemini",
                "vector_db_path": self.vector_db_path,
                "model_used": GEMINI_MODEL
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"結果已保存到: {output_path}")
        return output_path