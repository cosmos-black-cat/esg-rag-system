#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG RAG系統主運行腳本 - 使用 Google Gemini API
"""

import os
import sys
import csv
from datetime import datetime
from pathlib import Path
import argparse

# 添加src目錄到路徑
sys.path.append(str(Path(__file__).parent / "src"))

from esg_extractor import ESGDataExtractor
from preprocess import preprocess_documents
from config import *

def setup_system():
    """系統初始設置"""
    print("🚀 ESG RAG系統初始化 (使用 Google Gemini)")
    print("=" * 50)
    
    # 檢查必要目錄
    for dir_path in [DATA_PATH, RESULTS_PATH, os.path.dirname(VECTOR_DB_PATH)]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 目錄檢查: {dir_path}")
    
    # 檢查環境變數
    if not GOOGLE_API_KEY:
        print("❌ 錯誤: 請設置GOOGLE_API_KEY環境變數")
        print("方法1: 在.env文件中設置 GOOGLE_API_KEY=your_key")
        print("方法2: export GOOGLE_API_KEY=your_key")
        print("📋 獲取API Key: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"✅ Google API Key: {GOOGLE_API_KEY[:10]}...")
    print(f"✅ 使用模型: {GEMINI_MODEL}")
    print("=" * 50)
    return True

def check_and_preprocess():
    """檢查並預處理PDF文件"""
    
    # 檢查是否已有向量資料庫
    if os.path.exists(VECTOR_DB_PATH):
        print(f"✅ 找到現有向量資料庫: {VECTOR_DB_PATH}")
        return True
    
    # 檢查PDF文件
    data_dir = Path(DATA_PATH)
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {DATA_PATH} 目錄中找不到PDF文件")
        print("請將ESG報告PDF文件放入data目錄中")
        return False
    
    print(f"📄 找到PDF文件: {pdf_files[0]}")
    
    # 執行預處理
    try:
        print("🔄 開始預處理...")
        preprocess_documents(str(pdf_files[0]))
        print("✅ 預處理完成")
        return True
    except Exception as e:
        print(f"❌ 預處理失敗: {e}")
        return False

def extract_all_data():
    """提取所有ESG數據"""
    try:
        print("🔍 初始化ESG數據提取器...")
        extractor = ESGDataExtractor()
        
        print("📊 開始提取所有關鍵字數據...")
        results = extractor.extract_all_keywords()
        
        print("📋 生成摘要報告...")
        summary = extractor.generate_summary_report(results)
        
        # 保存結果
        json_path = extractor.save_results(results, summary)
        
        # 生成CSV報告
        csv_path = generate_csv_report(results, summary)
        
        # 打印摘要
        print_summary(summary)
        
        return results, summary, json_path, csv_path
        
    except Exception as e:
        print(f"❌ 數據提取失敗: {e}")
        return None, None, None, None

def generate_csv_report(results, summary):
    """生成CSV報告"""
    
    csv_filename = os.path.join(RESULTS_PATH, f"esg_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ["指標類別", "關鍵字", "提取值", "數據類型", "信心分數", "來源頁面", "來源文本"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for indicator, indicator_results in results.items():
            for result in indicator_results:
                writer.writerow({
                    "指標類別": result.indicator,
                    "關鍵字": result.keyword,
                    "提取值": result.value,
                    "數據類型": result.value_type,
                    "信心分數": f"{result.confidence:.2f}",
                    "來源頁面": result.page_info,
                    "來源文本": result.source_text[:100] + "..." if len(result.source_text) > 100 else result.source_text
                })
    
    print(f"📊 CSV報告已保存: {csv_filename}")
    return csv_filename

def print_summary(summary):
    """打印摘要結果"""
    print("\n" + "="*60)
    print("📊 ESG數據提取結果摘要 (使用 Google Gemini)")
    print("="*60)
    
    print(f"總關鍵字數量: {summary['total_keywords']}")
    print(f"成功提取: {summary['found_keywords']} ({summary['found_keywords']/summary['total_keywords']*100:.1f}%)")
    print(f"未找到數據: {summary['not_found_keywords']} ({summary['not_found_keywords']/summary['total_keywords']*100:.1f}%)")
    print(f"高信心結果: {summary['high_confidence_results']}")
    
    print("\n各指標詳細結果:")
    print("-" * 60)
    
    for indicator, indicator_summary in summary["indicators_summary"].items():
        print(f"\n📈 {indicator}")
        print(f"   成功率: {indicator_summary['found']}/{indicator_summary['total']} "
              f"({indicator_summary['found']/indicator_summary['total']*100:.1f}%)")
        
        # 顯示重要發現
        if indicator_summary['key_findings']:
            print("   🔍 重要發現:")
            for finding in indicator_summary['key_findings'][:3]:  # 只顯示前3個
                value_str = str(finding['value'])
                if len(value_str) > 30:
                    value_str = value_str[:30] + "..."
                print(f"      • {finding['keyword']}: {value_str} (信心度:{finding['confidence']:.2f})")
    
    print("\n" + "="*60)

def search_keyword(keyword):
    """搜尋特定關鍵字"""
    try:
        extractor = ESGDataExtractor()
        print(f"🔍 搜尋關鍵字: {keyword}")
        
        search_results = extractor.search_and_rerank(keyword, k=5)
        
        if not search_results:
            print("❌ 未找到相關內容")
            return
        
        print("📋 搜尋結果:")
        for i, (doc, score) in enumerate(search_results, 1):
            print(f"\n{i}. 相關度分數: {score:.3f}")
            print(f"   頁面: {doc.metadata.get('page', 'unknown')}")
            print(f"   內容預覽: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"❌ 搜尋失敗: {e}")

def test_gemini_connection():
    """測試Gemini API連接"""
    try:
        print("🧪 測試 Gemini API 連接...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        
        response = llm.invoke("請用中文回答：你好，測試連接是否正常？")
        print("✅ Gemini API 連接成功！")
        print(f"📝 測試回應: {response.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API 連接失敗: {e}")
        print("💡 請檢查:")
        print("   1. GOOGLE_API_KEY 是否正確設置")
        print("   2. 網路連接是否正常")
        print("   3. API Key 是否有效")
        return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='ESG RAG系統 (Google Gemini)')
    parser.add_argument('--action', choices=['setup', 'extract', 'search', 'test'], 
                       default='extract', help='執行的動作')
    parser.add_argument('--keyword', type=str, help='搜尋的關鍵字')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        # 只執行系統設置
        if setup_system():
            print("✅ 系統設置完成")
        return
    
    if args.action == 'test':
        # 測試Gemini連接
        if setup_system():
            test_gemini_connection()
        return
    
    # 系統初始化
    if not setup_system():
        return
    
    # 測試API連接
    if not test_gemini_connection():
        return
    
    # 檢查並預處理
    if not check_and_preprocess():
        return
    
    if args.action == 'extract':
        # 提取所有數據
        results, summary, json_path, csv_path = extract_all_data()
        if results:
            print(f"\n🎉 提取完成！")
            print(f"📁 JSON結果: {json_path}")
            print(f"📊 CSV報告: {csv_path}")
    
    elif args.action == 'search':
        # 搜尋特定關鍵字
        if not args.keyword:
            keyword = input("請輸入要搜尋的關鍵字: ").strip()
        else:
            keyword = args.keyword
        search_keyword(keyword)

def interactive_mode():
    """互動模式"""
    if not setup_system():
        return
    
    if not test_gemini_connection():
        return
    
    if not check_and_preprocess():
        return
    
    while True:
        print("\n" + "="*50)
        print("ESG RAG系統 - 互動模式 (Google Gemini)")
        print("="*50)
        print("1. 提取所有關鍵字數據")
        print("2. 搜尋特定關鍵字")
        print("3. 重新預處理PDF")
        print("4. 測試Gemini連接")
        print("5. 退出")
        
        choice = input("請選擇功能 (1-5): ").strip()
        
        if choice == "1":
            results, summary, json_path, csv_path = extract_all_data()
            if results:
                print(f"\n🎉 提取完成！")
                print(f"📁 詳細結果: {json_path}")
                print(f"📊 CSV報告: {csv_path}")
        
        elif choice == "2":
            keyword = input("請輸入要搜尋的關鍵字: ").strip()
            if keyword:
                search_keyword(keyword)
        
        elif choice == "3":
            # 重新預處理
            import shutil
            if os.path.exists(VECTOR_DB_PATH):
                shutil.rmtree(VECTOR_DB_PATH)
            check_and_preprocess()
        
        elif choice == "4":
            test_gemini_connection()
        
        elif choice == "5":
            print("👋 再見！")
            break
        
        else:
            print("❌ 無效選擇，請重試")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 沒有命令行參數，進入互動模式
        interactive_mode()
    else:
        # 有命令行參數，使用命令行模式
        main()