import json
import hashlib
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(str(Path(__file__).parent))
from config import *
from keywords_config import ESG_KEYWORDS_CONFIG, EXTRACTION_PROMPTS
from api_manager import GeminiAPIManager, GEMINI_API_KEYS

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
    explanation: str = ""
    api_key_used: str = ""  # 記錄使用的API key

@dataclass
class SimilarKeywordGroup:
    """相似關鍵字組"""
    keywords: List[str]
    similarity_score: float
    common_value: str
    confidence_avg: float

class MultiKeyESGDataExtractor:
    def __init__(self, vector_db_path: str = None):
        """初始化ESG數據提取器 - 支持多API key"""
        
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
        
        # 初始化多API key管理器
        print("初始化多API key Gemini管理器...")
        self.api_manager = GeminiAPIManager(
            api_keys=GEMINI_API_KEYS,
            model_name=GEMINI_MODEL
        )
        
        print("✅ 多API key ESG數據提取器初始化完成")

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
        try:
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
        except Exception as e:
            print(f"搜尋錯誤 ({query}): {e}")
            return []

    def extract_data_with_llm(self, keyword: str, context: str, data_type: str) -> ExtractionResult:
        """使用多API key Gemini提取特定數據"""
        
        # 選擇對應的提示模板
        prompt_template = EXTRACTION_PROMPTS.get(data_type, EXTRACTION_PROMPTS["number"])
        prompt = prompt_template.format(keyword=keyword, context=context)
        
        # 為Gemini優化prompt
        gemini_prompt = f"""
你是一個專業的ESG數據分析師。請仔細分析以下文本並提取相關信息。

{prompt}

重要提醒：
1. 必須給出一個明確的答案，不能回答"不知道"或"無法確定"
2. 如果文本中沒有直接提到該關鍵字，但有相關概念，請基於相關信息推斷
3. 如果完全沒有相關信息，請明確回答"未提及"
4. 請確保回答格式為有效的JSON

請以JSON格式回答，不要包含任何其他文字或解釋。
"""
        
        try:
            # 使用API管理器調用
            print(f"🤖 正在提取關鍵字: {keyword}")
            response = self.api_manager.invoke(gemini_prompt)
            
            # 清理響應內容，移除可能的markdown格式
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            result_json = json.loads(content)
            
            # 獲取當前使用的API key信息
            current_key = self.api_manager.api_keys[self.api_manager.current_key_index]
            
            return ExtractionResult(
                keyword=keyword,
                indicator="",  # 將在調用處設置
                value=result_json.get("value", "未提及"),
                value_type=data_type,
                confidence=result_json.get("confidence", 0.8),
                source_text=context[:200] + "..." if len(context) > 200 else context,
                page_info="",  # 將在調用處設置
                explanation=result_json.get("explanation", ""),
                api_key_used=current_key[:10] + "..."
            )
            
        except Exception as e:
            print(f"❌ LLM提取失敗 ({keyword}): {e}")
            # 即使LLM提取失敗，也要給出一個答案
            return ExtractionResult(
                keyword=keyword,
                indicator="",
                value="未提及",
                value_type=data_type,
                confidence=0.5,
                source_text="LLM提取過程中出現錯誤",
                page_info="",
                explanation=f"提取過程中出現錯誤: {str(e)}",
                api_key_used="error"
            )

    def extract_keyword_data(self, keyword: str, indicator: str, data_type: str) -> ExtractionResult:
        """對單個關鍵字進行數據提取，確保有答案"""
        
        # 1. 搜尋相關文檔
        search_results = self.search_and_rerank(keyword, k=3)
        
        if not search_results:
            # 沒有找到相關文檔，但還是要給出答案
            return ExtractionResult(
                keyword=keyword,
                indicator=indicator,
                value="報告書中沒有提到",
                value_type=data_type,
                confidence=1.0,
                source_text="未找到相關文檔",
                page_info="",
                explanation="在報告書中未找到與此關鍵字相關的內容",
                api_key_used="no_search"
            )
        
        # 2. 合併相關文檔的內容
        combined_context = ""
        page_info = []
        for doc, score in search_results:
            combined_context += doc.page_content + "\n\n"
            page_info.append(f"第{doc.metadata.get('page', 'unknown')}頁")
        
        # 3. 使用多API key Gemini提取數據
        result = self.extract_data_with_llm(keyword, combined_context, data_type)
        result.indicator = indicator
        result.page_info = ", ".join(page_info)
        
        return result

    def extract_all_keywords(self) -> Dict[str, List[ExtractionResult]]:
        """提取所有關鍵字的數據，使用多API key管理"""
        all_results = {}
        
        total_keywords = sum(len(config["keywords"]) for config in self.keywords_config.values())
        
        print(f"\n🔍 開始提取 {total_keywords} 個關鍵字的數據...")
        print(f"🔑 使用 {len(GEMINI_API_KEYS)} 個API key輪換")
        
        with tqdm(total=total_keywords, desc="提取ESG數據") as pbar:
            for indicator, config in self.keywords_config.items():
                print(f"\n📊 處理指標: {indicator}")
                indicator_results = []
                
                for keyword in config["keywords"]:
                    pbar.set_description(f"提取: {keyword}")
                    
                    # 對每個關鍵字進行提取
                    result = self.extract_keyword_data(keyword, indicator, config["type"])
                    indicator_results.append(result)
                    
                    # 顯示進度和使用的API key
                    status = "✓" if result.value != "未提及" and result.value != "報告書中沒有提到" else "○"
                    print(f"  {status} {keyword}: {result.value} (key: {result.api_key_used})")
                    
                    pbar.update(1)
                
                all_results[indicator] = indicator_results
                
                # 每完成一個指標，顯示API使用統計
                if len(indicator_results) > 5:  # 避免太頻繁的統計輸出
                    print(f"📈 已完成 {indicator}，API使用統計:")
                    self._print_brief_api_stats()
        
        return all_results

    def _print_brief_api_stats(self):
        """簡要打印API使用統計"""
        stats = self.api_manager.get_usage_statistics()
        total = stats["total_requests"]
        
        key_usage = []
        for key_name, key_stats in stats["keys_usage"].items():
            usage = key_stats["usage_count"]
            if usage > 0:
                key_usage.append(f"{key_name}:{usage}")
        
        print(f"   📊 總請求: {total} | " + " | ".join(key_usage))

    def find_similar_keywords(self, results: Dict[str, List[ExtractionResult]], 
                            top_n: int = 10) -> List[SimilarKeywordGroup]:
        """找出相似度最高的關鍵字組合"""
        print("\n🔍 分析關鍵字相似度...")
        
        # 收集所有結果
        all_results = []
        for indicator_results in results.values():
            all_results.extend(indicator_results)
        
        # 過濾出有效結果（非"未提及"）
        valid_results = [r for r in all_results 
                        if r.value not in ["未提及", "報告書中沒有提到"] 
                        and r.confidence > 0.6]
        
        if len(valid_results) < 2:
            return []
        
        # 創建關鍵字和值的配對
        keywords = [r.keyword for r in valid_results]
        values = [str(r.value) for r in valid_results]
        
        # 使用TF-IDF計算關鍵字相似度
        vectorizer = TfidfVectorizer()
        try:
            keyword_vectors = vectorizer.fit_transform(keywords)
            similarity_matrix = cosine_similarity(keyword_vectors)
        except:
            # 如果TF-IDF失敗，使用簡單的字符串相似度
            similarity_matrix = np.zeros((len(keywords), len(keywords)))
            for i in range(len(keywords)):
                for j in range(len(keywords)):
                    if i != j:
                        # 簡單的字符串相似度
                        k1, k2 = keywords[i].lower(), keywords[j].lower()
                        common_chars = len(set(k1) & set(k2))
                        total_chars = len(set(k1) | set(k2))
                        similarity_matrix[i][j] = common_chars / total_chars if total_chars > 0 else 0
        
        # 找出相似度最高的組合
        similar_groups = []
        processed_pairs = set()
        
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                if (i, j) in processed_pairs:
                    continue
                
                similarity = similarity_matrix[i][j]
                if similarity > 0.3:  # 相似度閾值
                    # 檢查是否有相同或相似的值
                    val1, val2 = values[i], values[j]
                    if val1 == val2 or self._values_similar(val1, val2):
                        group = SimilarKeywordGroup(
                            keywords=[keywords[i], keywords[j]],
                            similarity_score=similarity,
                            common_value=val1,
                            confidence_avg=(valid_results[i].confidence + valid_results[j].confidence) / 2
                        )
                        similar_groups.append(group)
                        processed_pairs.add((i, j))
        
        # 按相似度排序並返回前N個
        similar_groups.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_groups[:top_n]

    def _values_similar(self, val1: str, val2: str) -> bool:
        """判斷兩個值是否相似"""
        try:
            # 嘗試提取數字
            import re
            nums1 = re.findall(r'\d+(?:\.\d+)?', str(val1))
            nums2 = re.findall(r'\d+(?:\.\d+)?', str(val2))
            
            if nums1 and nums2:
                # 如果都有數字，比較第一個數字
                return abs(float(nums1[0]) - float(nums2[0])) < 0.1
        except:
            pass
        
        return str(val1).lower() == str(val2).lower()

    def generate_indicator_statistics(self, results: Dict[str, List[ExtractionResult]]) -> pd.DataFrame:
        """生成各指標統計"""
        print("\n📊 生成指標統計...")
        
        stats_data = []
        
        for indicator, indicator_results in results.items():
            total_keywords = len(indicator_results)
            found_keywords = len([r for r in indicator_results 
                                if r.value not in ["未提及", "報告書中沒有提到"]])
            not_found_keywords = total_keywords - found_keywords
            success_rate = (found_keywords / total_keywords * 100) if total_keywords > 0 else 0
            
            # 高信心結果
            high_confidence = len([r for r in indicator_results if r.confidence > 0.8])
            
            # 平均信心分數
            avg_confidence = np.mean([r.confidence for r in indicator_results])
            
            # 關鍵發現數量
            key_findings = len([r for r in indicator_results 
                              if r.confidence > 0.7 and r.value not in ["未提及", "報告書中沒有提到"]])
            
            # API key使用統計
            api_key_usage = {}
            for result in indicator_results:
                key = result.api_key_used
                api_key_usage[key] = api_key_usage.get(key, 0) + 1
            
            most_used_key = max(api_key_usage.items(), key=lambda x: x[1])[0] if api_key_usage else "unknown"
            
            stats_data.append({
                '指標名稱': indicator,
                '總關鍵字數': total_keywords,
                '成功提取數': found_keywords,
                '未找到數': not_found_keywords,
                '成功率(%)': round(success_rate, 1),
                '高信心結果數': high_confidence,
                '平均信心分數': round(avg_confidence, 3),
                '關鍵發現數': key_findings,
                '主要使用API': most_used_key
            })
        
        return pd.DataFrame(stats_data)

    def generate_excel_report(self, results: Dict[str, List[ExtractionResult]], 
                            output_path: str = None) -> str:
        """生成包含多個工作表的Excel報告"""
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(RESULTS_PATH, f"esg_multikey_report_{timestamp}.xlsx")
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n📊 生成Excel報告: {output_path}")
        
        # 創建Excel Writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # 工作表1: 完整提取結果
            print("  📋 生成工作表1: 完整提取結果")
            all_data = []
            for indicator, indicator_results in results.items():
                for result in indicator_results:
                    all_data.append({
                        '指標類別': result.indicator,
                        '關鍵字': result.keyword,
                        '提取值': str(result.value),
                        '數據類型': result.value_type,
                        '信心分數': round(result.confidence, 3),
                        '來源頁面': result.page_info,
                        '來源文本': result.source_text[:100] + "..." if len(result.source_text) > 100 else result.source_text,
                        '說明': result.explanation,
                        '使用的API Key': result.api_key_used
                    })
            
            df_all = pd.DataFrame(all_data)
            df_all.to_excel(writer, sheet_name='完整提取結果', index=False)
            
            # 工作表2: 相似關鍵字結果
            print("  🔗 生成工作表2: 相似關鍵字結果")
            similar_groups = self.find_similar_keywords(results)
            
            if similar_groups:
                similar_data = []
                for i, group in enumerate(similar_groups, 1):
                    for keyword in group.keywords:
                        # 找到對應的結果
                        for indicator_results in results.values():
                            for result in indicator_results:
                                if result.keyword == keyword:
                                    similar_data.append({
                                        '組別': f"相似組{i}",
                                        '相似度分數': round(group.similarity_score, 3),
                                        '共同值': group.common_value,
                                        '指標類別': result.indicator,
                                        '關鍵字': result.keyword,
                                        '提取值': str(result.value),
                                        '信心分數': round(result.confidence, 3),
                                        '來源頁面': result.page_info,
                                        '使用的API Key': result.api_key_used
                                    })
                                    break
                
                df_similar = pd.DataFrame(similar_data)
                df_similar.to_excel(writer, sheet_name='相似關鍵字結果', index=False)
            else:
                # 如果沒有相似關鍵字，創建空工作表
                pd.DataFrame({'說明': ['未找到相似度較高的關鍵字組合']}).to_excel(
                    writer, sheet_name='相似關鍵字結果', index=False)
            
            # 工作表3: 各指標統計
            print("  📈 生成工作表3: 各指標統計")
            df_stats = self.generate_indicator_statistics(results)
            df_stats.to_excel(writer, sheet_name='各指標統計', index=False)
            
            # 工作表4: 摘要統計
            print("  📊 生成工作表4: 摘要統計")
            summary = self.generate_summary_report(results)
            api_stats = self.api_manager.get_usage_statistics()
            
            summary_data = [
                {'項目': '總關鍵字數量', '數值': summary['total_keywords']},
                {'項目': '成功提取數量', '數值': summary['found_keywords']},
                {'項目': '未找到數量', '數值': summary['not_found_keywords']},
                {'項目': '整體成功率(%)', '數值': round(summary['found_keywords']/summary['total_keywords']*100, 1)},
                {'項目': '高信心結果數', '數值': summary['high_confidence_results']},
                {'項目': '相似關鍵字組數', '數值': len(similar_groups)},
                {'項目': '總API請求次數', '數值': api_stats['total_requests']},
                {'項目': '使用的API Key數量', '數值': len(GEMINI_API_KEYS)},
                {'項目': '生成時間', '數值': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            ]
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='摘要統計', index=False)
            
            # 工作表5: API使用統計
            print("  🔑 生成工作表5: API使用統計")
            api_data = []
            for key_name, key_stats in api_stats["keys_usage"].items():
                api_data.append({
                    'API Key': key_name,
                    'Key預覽': key_stats['key_preview'],
                    '使用次數': key_stats['usage_count'],
                    '使用比例(%)': round(key_stats['usage_percentage'], 1),
                    '狀態': '冷卻中' if key_stats['is_cooling'] else '可用'
                })
            
            df_api = pd.DataFrame(api_data)
            df_api.to_excel(writer, sheet_name='API使用統計', index=False)
        
        print(f"✅ Excel報告生成完成: {output_path}")
        return output_path

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
                
                if result.value in ["未提及", "報告書中沒有提到"]:
                    summary["not_found_keywords"] += 1
                    indicator_summary["not_found"] += 1
                else:
                    summary["found_keywords"] += 1
                    indicator_summary["found"] += 1
                    
                    if result.confidence > 0.7:
                        summary["high_confidence_results"] += 1
                        indicator_summary["high_confidence"] += 1
                    
                    # 收集重要發現
                    if result.confidence > 0.6 and result.value not in ["未提及", "報告書中沒有提到"]:
                        indicator_summary["key_findings"].append({
                            "keyword": result.keyword,
                            "value": result.value,
                            "confidence": result.confidence
                        })
            
            summary["indicators_summary"][indicator] = indicator_summary
        
        return summary

    def print_summary(self, summary: Dict, similar_groups: List[SimilarKeywordGroup]):
        """打印摘要結果和API使用統計"""
        print("\n" + "="*60)
        print("📊 ESG數據提取結果摘要 (多API Key版本)")
        print("="*60)
        
        print(f"總關鍵字數量: {summary['total_keywords']}")
        print(f"成功提取: {summary['found_keywords']} ({summary['found_keywords']/summary['total_keywords']*100:.1f}%)")
        print(f"未找到數據: {summary['not_found_keywords']} ({summary['not_found_keywords']/summary['total_keywords']*100:.1f}%)")
        print(f"高信心結果: {summary['high_confidence_results']}")
        print(f"相似關鍵字組: {len(similar_groups)}")
        
        # 顯示API使用統計
        self.api_manager.print_usage_statistics()
        
        print("\n各指標詳細結果:")
        print("-" * 60)
        
        for indicator, indicator_summary in summary["indicators_summary"].items():
            print(f"\n📈 {indicator}")
            print(f"   成功率: {indicator_summary['found']}/{indicator_summary['total']} "
                  f"({indicator_summary['found']/indicator_summary['total']*100:.1f}%)")
            
            # 顯示重要發現
            if indicator_summary['key_findings']:
                print("   🔍 重要發現:")
                for finding in indicator_summary['key_findings'][:3]:
                    value_str = str(finding['value'])
                    if len(value_str) > 30:
                        value_str = value_str[:30] + "..."
                    print(f"      • {finding['keyword']}: {value_str} (信心度:{finding['confidence']:.2f})")
        
        if similar_groups:
            print(f"\n🔗 相似關鍵字組合 (前{min(5, len(similar_groups))}組):")
            print("-" * 60)
            for i, group in enumerate(similar_groups[:5], 1):
                print(f"{i}. 相似度: {group.similarity_score:.3f} | 共同值: {group.common_value}")
                print(f"   關鍵字: {', '.join(group.keywords)}")
        
        print("\n" + "="*60)

def main():
    """主函數 - 使用多API key生成Excel報告"""
    
    try:
        print("🚀 ESG數據提取器 - 多API Key版本")
        print("=" * 60)
        
        # 初始化提取器
        extractor = MultiKeyESGDataExtractor()
        
        # 提取所有關鍵字數據
        results = extractor.extract_all_keywords()
        
        # 生成摘要
        summary = extractor.generate_summary_report(results)
        
        # 找出相似關鍵字
        similar_groups = extractor.find_similar_keywords(results)
        
        # 生成Excel報告
        excel_path = extractor.generate_excel_report(results)
        
        # 打印摘要
        extractor.print_summary(summary, similar_groups)
        
        print(f"\n🎉 提取完成！")
        print(f"📊 Excel報告: {excel_path}")
        print(f"📋 包含工作表:")
        print(f"   • 工作表1: 完整提取結果")
        print(f"   • 工作表2: 相似關鍵字結果")
        print(f"   • 工作表3: 各指標統計")
        print(f"   • 工作表4: 摘要統計")
        print(f"   • 工作表5: API使用統計")
        
        return results, summary, excel_path
        
    except Exception as e:
        print(f"❌ 提取過程中出現錯誤: {e}")
        return None, None, None

if __name__ == "__main__":
    main()