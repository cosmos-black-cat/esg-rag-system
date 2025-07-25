#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG RAG系統主運行腳本 - 多API Key版本
支持API key輪換和智能等待機制
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加src目錄到路徑
sys.path.append(str(Path(__file__).parent / "src"))

def check_dependencies():
    """檢查依賴包"""
    required_packages = {
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',
        'sklearn': 'scikit-learn',
        'google.generativeai': 'google-generativeai',
        'google.api_core': 'google-api-core'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("❌ 缺少必要的依賴包:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n請安裝缺少的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_api_keys():
    """檢查API keys配置"""
    try:
        from api_manager import GEMINI_API_KEYS
        
        if not GEMINI_API_KEYS or len(GEMINI_API_KEYS) == 0:
            print("❌ 錯誤: 沒有配置Gemini API keys")
            return False
        
        # 檢查API key格式
        valid_keys = []
        for i, key in enumerate(GEMINI_API_KEYS):
            if key and len(key) > 20 and key.startswith('AIza'):
                valid_keys.append(key)
            else:
                print(f"⚠️  警告: API key {i+1} 格式可能不正確")
        
        if len(valid_keys) == 0:
            print("❌ 錯誤: 沒有有效的API keys")
            return False
        
        print(f"✅ 找到 {len(valid_keys)} 個有效的API keys")
        return True
        
    except ImportError:
        print("❌ 錯誤: 無法導入API管理器")
        return False

def check_environment():
    """檢查環境和依賴"""
    print("🔍 檢查系統環境...")
    
    # 檢查Python包
    if not check_dependencies():
        return False
    
    # 檢查API keys
    if not check_api_keys():
        return False
    
    try:
        from config import VECTOR_DB_PATH, DATA_PATH, RESULTS_PATH
        
        # 檢查向量資料庫
        if not os.path.exists(VECTOR_DB_PATH):
            print(f"❌ 錯誤: 向量資料庫不存在: {VECTOR_DB_PATH}")
            print("請先運行預處理步驟或確保PDF文件已處理")
            return False
        
        # 檢查結果目錄
        os.makedirs(RESULTS_PATH, exist_ok=True)
        
        print("✅ 環境檢查通過")
        return True
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("請確保已安裝所有必要的依賴包")
        return False

def test_api_keys():
    """測試API keys是否可用"""
    print("🧪 測試API keys...")
    
    try:
        from api_manager import GeminiAPIManager, GEMINI_API_KEYS
        
        # 創建API管理器
        api_manager = GeminiAPIManager(GEMINI_API_KEYS)
        
        # 測試一個簡單的請求
        test_prompt = "請用中文回答：你好，這是一個測試請求。"
        
        try:
            response = api_manager.invoke(test_prompt)
            print("✅ API測試成功")
            print(f"📝 測試響應: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"❌ API測試失敗: {e}")
            return False
            
    except Exception as e:
        print(f"❌ API測試初始化失敗: {e}")
        return False

def run_multi_key_extraction():
    """運行多API key ESG數據提取"""
    print("🚀 開始ESG數據提取 - 多API Key版本")
    print("=" * 60)
    
    try:
        from esg_extractor import MultiKeyESGDataExtractor
        
        # 初始化提取器
        print("📱 初始化多API key ESG數據提取器...")
        extractor = MultiKeyESGDataExtractor()
        
        # 提取所有關鍵字數據
        print("🔍 開始提取數據...")
        results = extractor.extract_all_keywords()
        
        # 生成摘要
        print("📊 生成摘要報告...")
        summary = extractor.generate_summary_report(results)
        
        # 找出相似關鍵字
        print("🔗 分析相似關鍵字...")
        similar_groups = extractor.find_similar_keywords(results)
        
        # 生成Excel報告
        print("📊 生成Excel多工作表報告...")
        excel_path = extractor.generate_excel_report(results)
        
        # 打印摘要
        extractor.print_summary(summary, similar_groups)
        
        print(f"\n🎉 數據提取完成！")
        print(f"📊 Excel報告: {excel_path}")
        print(f"📋 包含以下工作表:")
        print(f"   • 工作表1: 完整提取結果 (所有{summary['total_keywords']}個關鍵字)")
        print(f"   • 工作表2: 相似關鍵字結果 ({len(similar_groups)}組相似關鍵字)")
        print(f"   • 工作表3: 各指標統計 (6個指標的詳細統計)")
        print(f"   • 工作表4: 摘要統計 (整體統計信息)")
        print(f"   • 工作表5: API使用統計 (多key使用情況)")
        print(f"⏰ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return excel_path, summary, similar_groups
        
    except Exception as e:
        print(f"❌ 提取過程中出現錯誤: {e}")
        print("請檢查錯誤信息並重試")
        return None, None, None

def analyze_multi_key_results(excel_path):
    """分析多API key Excel結果"""
    try:
        import pandas as pd
        
        print(f"\n📈 分析Excel報告: {excel_path}")
        print("=" * 60)
        
        # 讀取各個工作表
        sheets = pd.read_excel(excel_path, sheet_name=None)
        
        for sheet_name, df in sheets.items():
            print(f"\n📋 {sheet_name}:")
            print(f"   行數: {len(df)}")
            print(f"   列數: {len(df.columns) if not df.empty else 0}")
            
            if sheet_name == "完整提取結果":
                # 分析成功率
                success_count = len(df[df['提取值'] != '未提及'])
                total_count = len(df)
                print(f"   成功提取: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
                
                # 分析API key使用分佈
                if '使用的API Key' in df.columns:
                    api_usage = df['使用的API Key'].value_counts()
                    print("   API Key使用分佈:")
                    for api_key, count in api_usage.head(3).items():
                        print(f"     • {api_key}: {count} 次")
            
            elif sheet_name == "API使用統計":
                if not df.empty and '使用次數' in df.columns:
                    total_requests = df['使用次數'].sum()
                    max_usage = df['使用次數'].max()
                    min_usage = df['使用次數'].min()
                    print(f"   總請求: {total_requests}")
                    print(f"   最高使用: {max_usage} 次")
                    print(f"   最低使用: {min_usage} 次")
                    print(f"   負載均衡度: {min_usage/max_usage*100:.1f}%" if max_usage > 0 else "   負載均衡度: N/A")
            
            elif sheet_name == "相似關鍵字結果":
                if not df.empty and '組別' in df.columns:
                    group_count = df['組別'].nunique()
                    print(f"   相似組數: {group_count}")
                    if group_count > 0:
                        avg_similarity = df['相似度分數'].mean()
                        print(f"   平均相似度: {avg_similarity:.3f}")
            
            elif sheet_name == "各指標統計":
                if not df.empty and '成功率(%)' in df.columns:
                    avg_success_rate = df['成功率(%)'].mean()
                    print(f"   平均成功率: {avg_success_rate:.1f}%")
                    best_indicator = df.loc[df['成功率(%)'].idxmax(), '指標名稱']
                    best_rate = df['成功率(%)'].max()
                    print(f"   最佳指標: {best_indicator} ({best_rate:.1f}%)")
        
        print("\n✅ Excel報告分析完成")
        
    except Exception as e:
        print(f"❌ 分析Excel報告失敗: {e}")

def show_latest_results():
    """顯示最新結果"""
    try:
        from config import RESULTS_PATH
        import glob
        
        # 找到最新的Excel文件
        excel_files = glob.glob(os.path.join(RESULTS_PATH, "esg_multikey_report_*.xlsx"))
        if excel_files:
            latest_excel = max(excel_files, key=os.path.getctime)
            print(f"\n📄 最新Excel報告: {latest_excel}")
            analyze_multi_key_results(latest_excel)
        else:
            print("❌ 未找到Excel報告文件，請先運行數據提取")
            
    except Exception as e:
        print(f"❌ 查看結果失敗: {e}")

def show_api_key_config():
    """顯示API key配置信息"""
    try:
        from api_manager import GEMINI_API_KEYS
        
        print("\n🔑 API Key配置信息:")
        print("=" * 40)
        print(f"配置的API Key數量: {len(GEMINI_API_KEYS)}")
        
        for i, key in enumerate(GEMINI_API_KEYS, 1):
            print(f"Key {i}: {key[:10]}...{key[-4:]}")
        
        print("\n💡 多API Key機制說明:")
        print("• 系統會自動輪換使用不同的API key")
        print("• 當某個key達到限制時，自動切換到下一個")
        print("• 如果所有key都達到限制，系統會等待10分鐘")
        print("• 每個請求之間有1秒的間隔以避免過於頻繁")
        
    except Exception as e:
        print(f"❌ 無法讀取API key配置: {e}")

def interactive_menu():
    """互動式選單"""
    while True:
        print("\n" + "="*60)
        print("🏢 ESG數據提取系統 - 多API Key版本")
        print("="*60)
        print("1. 運行完整數據提取 (多API key)")
        print("2. 檢查系統環境")
        print("3. 測試API keys")
        print("4. 查看最新Excel報告")
        print("5. 查看API key配置")
        print("6. 安裝依賴說明")
        print("7. 退出")
        
        choice = input("\n請選擇功能 (1-7): ").strip()
        
        if choice == "1":
            if check_environment():
                excel_path, summary, similar_groups = run_multi_key_extraction()
                if excel_path:
                    analyze_multi_key_results(excel_path)
        
        elif choice == "2":
            check_environment()
        
        elif choice == "3":
            if check_dependencies():
                test_api_keys()
        
        elif choice == "4":
            show_latest_results()
        
        elif choice == "5":
            show_api_key_config()
        
        elif choice == "6":
            print("\n📦 安裝依賴說明:")
            print("=" * 40)
            print("本系統需要以下Python包:")
            print("• pandas - 數據處理")
            print("• openpyxl - Excel文件操作")
            print("• scikit-learn - 相似度計算")
            print("• google-generativeai - Gemini API")
            print("• google-api-core - Google API核心庫")
            print("\n安裝命令:")
            print("pip install pandas openpyxl scikit-learn google-generativeai google-api-core")
            print("\n或者安裝所有依賴:")
            print("pip install -r requirements.txt")
        
        elif choice == "7":
            print("👋 感謝使用ESG數據提取系統！")
            break
        
        else:
            print("❌ 無效選擇，請重試")

def main():
    """主函數"""
    print("🏢 ESG數據提取系統 - 多API Key版本 v4.0")
    print("支持API key輪換和智能等待機制")
    print("=" * 60)
    
    # 檢查命令行參數
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto":
            # 自動運行模式
            if check_environment():
                excel_path, summary, similar_groups = run_multi_key_extraction()
                if excel_path:
                    analyze_multi_key_results(excel_path)
        elif sys.argv[1] == "--check":
            # 僅檢查環境
            check_environment()
        elif sys.argv[1] == "--test":
            # 測試API keys
            if check_dependencies():
                test_api_keys()
        elif sys.argv[1] == "--install":
            # 顯示安裝說明
            print("📦 請安裝以下依賴包:")
            print("pip install pandas openpyxl scikit-learn google-generativeai google-api-core")
        else:
            print("用法:")
            print("  python multi_key_main.py           # 互動模式")
            print("  python multi_key_main.py --auto    # 自動運行")
            print("  python multi_key_main.py --check   # 檢查環境")
            print("  python multi_key_main.py --test    # 測試API keys")
            print("  python multi_key_main.py --install # 安裝說明")
    else:
        # 互動模式
        interactive_menu()

if __name__ == "__main__":
    main()