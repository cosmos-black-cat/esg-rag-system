#!/usr/bin/env python3
# accuracy_evaluator.py - 評估和改進提取準確性

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

sys.path.append(str(Path(__file__).parent / "src"))

from esg_extractor import ESGDataExtractor
from config import *

class AccuracyEvaluator:
    def __init__(self):
        """初始化準確性評估器"""
        self.extractor = ESGDataExtractor()
        self.evaluation_results = []
    
    def create_test_cases(self) -> List[Dict]:
        """創建測試用例"""
        test_cases = [
            {
                "keyword": "再生材料使用比例",
                "context": "本公司2023年再生材料使用比例達到25.5%，較前年提升3個百分點，預計2024年將達到30%。",
                "expected_value": "25.5%",
                "expected_type": "percentage"
            },
            {
                "keyword": "耐用性提升",
                "context": "通過改良材料配方，產品耐用性顯著提升，平均使用壽命從5年延長至8年，提升幅度達60%。",
                "expected_value": "是",
                "expected_type": "boolean_or_number"
            },
            {
                "keyword": "每公斤成本",
                "context": "採用再生材料後，每公斤原料成本從50元降至35元，節省成本30%，年度節約金額達500萬元。",
                "expected_value": "35元",
                "expected_type": "number"
            },
            {
                "keyword": "循環次數",
                "context": "該材料可重複循環利用5次，每次循環後品質保持率達90%以上，符合循環經濟原則。",
                "expected_value": "5次",
                "expected_type": "number"
            },
            {
                "keyword": "碳足跡比較",
                "context": "與原生材料相比，再生材料的碳足跡減少40%，相當於每噸減少200kg CO2排放。",
                "expected_value": "減少40%",
                "expected_type": "number"
            },
            {
                "keyword": "不存在的指標",
                "context": "本報告討論了公司的財務表現，總收入增長15%，員工數量達到500人。",
                "expected_value": "報告書中沒有提到",
                "expected_type": "not_found"
            }
        ]
        return test_cases
    
    def evaluate_extraction(self, test_cases: List[Dict]) -> Dict:
        """評估提取準確性"""
        print("🧪 開始準確性評估...")
        print("=" * 50)
        
        results = {
            "total_tests": len(test_cases),
            "correct_extractions": 0,
            "partial_matches": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "detailed_results": []
        }
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 測試用例 {i}: {test_case['keyword']}")
            
            # 執行提取
            extraction_result = self.extractor.extract_data_with_llm(
                test_case["keyword"],
                test_case["context"],
                test_case["expected_type"]
            )
            
            # 評估結果
            evaluation = self._evaluate_single_result(
                extraction_result, test_case
            )
            
            results["detailed_results"].append(evaluation)
            
            # 統計結果
            if evaluation["match_type"] == "exact":
                results["correct_extractions"] += 1
                print(f"✅ 準確提取: {extraction_result.value}")
            elif evaluation["match_type"] == "partial":
                results["partial_matches"] += 1
                print(f"⚠️  部分匹配: {extraction_result.value} (期望: {test_case['expected_value']})")
            elif evaluation["match_type"] == "false_positive":
                results["false_positives"] += 1
                print(f"❌ 誤報: {extraction_result.value} (期望: {test_case['expected_value']})")
            else:
                results["false_negatives"] += 1
                print(f"❌ 漏報: {extraction_result.value} (期望: {test_case['expected_value']})")
            
            print(f"   信心分數: {extraction_result.confidence:.2f}")
        
        # 計算整體準確率
        accuracy = results["correct_extractions"] / results["total_tests"]
        results["accuracy"] = accuracy
        
        return results
    
    def _evaluate_single_result(self, extraction_result, test_case) -> Dict:
        """評估單個提取結果"""
        extracted_value = str(extraction_result.value).strip()
        expected_value = str(test_case["expected_value"]).strip()
        
        evaluation = {
            "keyword": test_case["keyword"],
            "extracted": extracted_value,
            "expected": expected_value,
            "confidence": extraction_result.confidence,
            "match_type": "false_negative"
        }
        
        # 精確匹配
        if extracted_value == expected_value:
            evaluation["match_type"] = "exact"
            return evaluation
        
        # 部分匹配檢查
        if self._is_partial_match(extracted_value, expected_value):
            evaluation["match_type"] = "partial"
            return evaluation
        
        # 檢查是否為誤報
        if expected_value in ["報告書中沒有提到", "未提及"] and extracted_value not in ["報告書中沒有提到", "未提及", "提取失敗"]:
            evaluation["match_type"] = "false_positive"
            return evaluation
        
        return evaluation
    
    def _is_partial_match(self, extracted: str, expected: str) -> bool:
        """檢查是否為部分匹配"""
        import re
        
        # 提取數字進行比較
        extracted_nums = re.findall(r'\d+\.?\d*', extracted)
        expected_nums = re.findall(r'\d+\.?\d*', expected)
        
        if extracted_nums and expected_nums:
            try:
                extracted_num = float(extracted_nums[0])
                expected_num = float(expected_nums[0])
                # 如果數字相近（差異小於10%），認為是部分匹配
                return abs(extracted_num - expected_num) / expected_num < 0.1
            except ValueError:
                pass
        
        # 文字部分匹配
        if "是" in extracted and "是" in expected:
            return True
        if "否" in extracted and "否" in expected:
            return True
        
        return False
    
    def generate_improvement_suggestions(self, results: Dict) -> List[str]:
        """生成改進建議"""
        suggestions = []
        
        accuracy = results["accuracy"]
        
        if accuracy < 0.7:
            suggestions.append("🔧 整體準確率較低，建議:")
            suggestions.append("   - 調整prompt模板，增加更多範例")
            suggestions.append("   - 提高信心分數閾值")
            suggestions.append("   - 增加數據驗證規則")
        
        if results["false_positives"] > 2:
            suggestions.append("⚠️  誤報較多，建議:")
            suggestions.append("   - 加強關鍵字相關性檢查")
            suggestions.append("   - 降低低相關性數據的信心分數")
        
        if results["false_negatives"] > 2:
            suggestions.append("❌ 漏報較多，建議:")
            suggestions.append("   - 擴展同義詞詞庫")
            suggestions.append("   - 改進文本預處理流程")
            suggestions.append("   - 增加多次嘗試機制")
        
        if results["partial_matches"] > 1:
            suggestions.append("📝 部分匹配較多，建議:")
            suggestions.append("   - 優化數據格式統一化")
            suggestions.append("   - 增加單位標準化處理")
        
        return suggestions
    
    def run_evaluation(self):
        """運行完整評估"""
        print("🚀 ESG數據提取準確性評估")
        print("=" * 50)
        
        # 創建測試用例
        test_cases = self.create_test_cases()
        
        # 執行評估
        results = self.evaluate_extraction(test_cases)
        
        # 打印結果摘要
        print("\n📊 評估結果摘要:")
        print("=" * 30)
        print(f"總測試數: {results['total_tests']}")
        print(f"正確提取: {results['correct_extractions']}")
        print(f"部分匹配: {results['partial_matches']}")
        print(f"誤報: {results['false_positives']}")
        print(f"漏報: {results['false_negatives']}")
        print(f"整體準確率: {results['accuracy']:.1%}")
        
        # 生成改進建議
        suggestions = self.generate_improvement_suggestions(results)
        if suggestions:
            print("\n💡 改進建議:")
            for suggestion in suggestions:
                print(suggestion)
        
        # 保存詳細結果
        output_path = os.path.join(RESULTS_PATH, f"accuracy_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 詳細結果已保存: {output_path}")
        
        return results

def main():
    """主函數"""
    evaluator = AccuracyEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()