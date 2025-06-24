#!/usr/bin/env python3
# test_improvements.py - 快速測試改進效果

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

def test_single_extraction():
    """測試單個關鍵字提取"""
    from esg_extractor import ESGDataExtractor
    
    # 初始化提取器
    try:
        extractor = ESGDataExtractor()
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        print("請確保:")
        print("1. 已設置 GOOGLE_API_KEY")
        print("2. 已運行 preprocess.py 建立向量資料庫")
        print("3. 已安裝所有依賴包")
        return False
    
    # 測試案例
    test_cases = [
        {
            "keyword": "再生材料使用比例",
            "context": "本公司致力於提升再生材料使用比例，2023年達到25.5%，較2022年的20.3%提升了5.2個百分點。預計2024年目標為30%。",
            "expected": "25.5%"
        },
        {
            "keyword": "耐用性提升",
            "context": "通過技術改良，產品耐用性顯著提升，平均使用壽命從原來的5年延長到8年，客戶滿意度大幅改善。",
            "expected": "是"
        },
        {
            "keyword": "循環次數",
            "context": "該包裝材料經測試可循環利用多達6次，每次循環後仍保持良好的物理特性和外觀品質。",
            "expected": "6次"
        }
    ]
    
    print("🧪 測試改進後的提取準確性")
    print("=" * 40)
    
    success_count = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 測試 {i}: {test['keyword']}")
        print(f"文本: {test['context'][:50]}...")
        print(f"期望結果: {test['expected']}")
        
        try:
            # 執行提取
            result = extractor.extract_data_with_llm(
                test["keyword"],
                test["context"],
                "percentage" if "%" in test["expected"] else "boolean_or_number" if test["expected"] in ["是", "否"] else "number"
            )
            
            print(f"提取結果: {result.value}")
            print(f"信心分數: {result.confidence:.2f}")
            
            # 簡單匹配檢查
            if str(result.value).strip() == test["expected"]:
                print("✅ 完全匹配!")
                success_count += 1
            elif test["expected"] in str(result.value) or str(result.value) in test["expected"]:
                print("⚠️  部分匹配")
                success_count += 0.5
            else:
                print("❌ 不匹配")
            
        except Exception as e:
            print(f"❌ 提取失敗: {e}")
    
    print(f"\n📊 測試結果: {success_count}/{len(test_cases)} = {success_count/len(test_cases):.1%}")
    
    if success_count >= len(test_cases) * 0.8:
        print("🎉 提取效果良好！")
        return True
    else:
        print("⚠️  提取效果需要進一步改善")
        return False

def test_search_relevance():
    """測試搜索相關性"""
    from esg_extractor import ESGDataExtractor
    
    print("\n🔍 測試搜索相關性")
    print("=" * 40)
    
    try:
        extractor = ESGDataExtractor()
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False
    
    test_keywords = [
        "再生材料使用比例",
        "產品壽命",
        "循環次數",
        "碳足跡比較"
    ]
    
    for keyword in test_keywords:
        print(f"\n🔎 搜索: {keyword}")
        
        try:
            search_results = extractor.search_and_rerank(keyword, k=3)
            
            if search_results:
                print(f"找到 {len(search_results)} 個相關文檔")
                for i, (doc, score) in enumerate(search_results, 1):
                    print(f"  {i}. 相關度: {score:.3f}")
                    print(f"     內容預覽: {doc.page_content[:80]}...")
            else:
                print("❌ 未找到相關文檔")
                
        except Exception as e:
            print(f"❌ 搜索失敗: {e}")
    
    return True

def run_quick_accuracy_test():
    """運行快速準確性測試"""
    try:
        from accuracy_evaluator import AccuracyEvaluator
        
        print("\n🎯 運行完整準確性評估")
        print("=" * 40)
        
        evaluator = AccuracyEvaluator()
        results = evaluator.run_evaluation()
        
        return results["accuracy"] > 0.7
        
    except Exception as e:
        print(f"⚠️  無法運行完整評估: {e}")
        return None

def main():
    """主函數"""
    print("🚀 ESG RAG系統改進效果測試")
    print("=" * 50)
    
    # 測試1: 單個提取
    extraction_ok = test_single_extraction()
    
    # 測試2: 搜索相關性  
    search_ok = test_search_relevance()
    
    # 測試3: 完整準確性評估（可選）
    print("\n" + "=" * 50)
    run_full_test = input("是否運行完整準確性評估? (y/n): ").lower().strip()
    
    if run_full_test == 'y':
        accuracy_ok = run_quick_accuracy_test()
    else:
        accuracy_ok = None
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試總結:")
    print("=" * 50)
    
    if extraction_ok:
        print("✅ 關鍵字提取: 良好")
    else:
        print("❌ 關鍵字提取: 需要改善")
    
    if search_ok:
        print("✅ 搜索功能: 正常")
    else:
        print("❌ 搜索功能: 有問題")
    
    if accuracy_ok is not None:
        if accuracy_ok:
            print("✅ 整體準確性: 良好")
        else:
            print("❌ 整體準確性: 需要改善")
    
    # 改善建議
    if not extraction_ok or not search_ok:
        print("\n💡 改善建議:")
        if not extraction_ok:
            print("- 檢查 prompt 模板是否正確載入")
            print("- 確認 Gemini API 連接穩定")
            print("- 調整信心分數閾值")
        if not search_ok:
            print("- 檢查向量資料庫是否正確建立")
            print("- 確認PDF預處理是否成功")
    else:
        print("\n🎉 系統運行良好！可以開始正式使用。")

if __name__ == "__main__":
    main()