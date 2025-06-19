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
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(str(Path(__file__).parent))
from config import *
from keywords_config import ESG_KEYWORDS_CONFIG, EXTRACTION_PROMPTS

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
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            convert_system_message_to_human=True  # Gemini需要這個設置
        )
        
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
        
        # 選擇對應的提示模板
        prompt_template = EXTRACTION_PROMPTS.get(data_type, EXTRACTION_PROMPTS["number"])
        prompt = prompt_template.format(keyword=keyword, context=context)
        
        try:
            # 為Gemini優化prompt
            gemini_prompt = f"""
你是一個專業的ESG數據分析師。請仔細分析以下文本並提取相關信息。

{prompt}

請確保回答格式為有效的JSON，不要包含任何其他文字或解釋。
"""
            
            response = self.llm.invoke(gemini_prompt)
            
            # 清理響應內容，移除可能的markdown格式
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            result_json = json.loads(content)
            
            return ExtractionResult(
                keyword=keyword,
                indicator="",  # 將在調用處設置
                value=result_json.get("value"),
                value_type=data_type,
                confidence=result_json.get("confidence", 0.0),
                source_text=context[:200] + "..." if len(context) > 200 else context,
                page_info=""  # 將在調用處設置
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