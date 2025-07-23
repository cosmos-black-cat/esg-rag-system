#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG RAG系統主運行腳本 - Excel多工作表版本
生成包含完整結果、相似關鍵字、指標統計的Excel報告
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
        'sklearn': 'scikit-learn'
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

def check_environment():
    """檢查環境和依賴"""
    print("🔍 檢查系統環境...")
    
    # 檢查Python包
    if not check_dependencies():
        return False
    
    try:
        from config import GOOGLE_API_KEY, GEMINI_MODEL, VECTOR_DB_PATH, DATA_PATH, RESULTS_PATH
        
        # 檢查API Key
        if not GOOGLE_API_KEY:
            print("❌ 錯誤: GOOGLE_API_KEY 未設置")
            print("請在 src/.env 文件中設置您的 Google API Key")
            return False
        
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

def run_excel_extraction():
    """運行ESG數據提取並生成Excel報告"""
    print("🚀 開始ESG數據提取 - Excel多工作表版本")
    print("=" * 60)
    
    try:
        from esg_extractor import ESGDataExtractor
        
        # 初始化提取器
        print("📱 初始化ESG數據提取器...")
        extractor = ESGDataExtractor()
        
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
        print(f"⏰ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return excel_path, summary, similar_groups
        
    except Exception as e:
        print(f"❌ 提取過程中出現錯誤: {e}")
        print("請檢查錯誤信息並重試")
        return None, None, None

def analyze_excel_results(excel_path):
    """分析Excel結果"""
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
                
                # 分析各指標
                indicator_stats = df.groupby('指標類別').agg({
                    '關鍵字': 'count',
                    '提取值': lambda x: (x != '未提及').sum()
                }).rename(columns={'關鍵字': '總數', '提取值': '成功數'})
                
                print("   各指標成功率:")
                for indicator, stats in indicator_stats.iterrows():
                    rate = stats['成功數'] / stats['總數'] * 100
                    print(f"     • {indicator}: {stats['成功數']}/{stats['總數']} ({rate:.1f}%)")
            
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
        excel_files = glob.glob(os.path.join(RESULTS_PATH, "esg_comprehensive_report_*.xlsx"))
        if excel_files:
            latest_excel = max(excel_files, key=os.path.getctime)
            print(f"\n📄 最新Excel報告: {latest_excel}")
            analyze_excel_results(latest_excel)
        else:
            print("❌ 未找到Excel報告文件，請先運行數據提取")
            
    except Exception as e:
        print(f"❌ 查看結果失敗: {e}")

def interactive_menu():
    """互動式選單"""
    while True:
        print("\n" + "="*60)
        print("🏢 ESG數據提取系統 - Excel多工作表版本")
        print("="*60)
        print("1. 運行完整數據提取 (生成Excel報告)")
        print("2. 檢查系統環境")
        print("3. 查看最新Excel報告")
        print("4. 安裝依賴說明")
        print("5. 退出")
        
        choice = input("\n請選擇功能 (1-5): ").strip()
        
        if choice == "1":
            if check_environment():
                excel_path, summary, similar_groups = run_excel_extraction()
                if excel_path:
                    analyze_excel_results(excel_path)
        
        elif choice == "2":
            check_environment()
        
        elif choice == "3":
            show_latest_results()
        
        elif choice == "4":
            print("\n📦 安裝依賴說明:")
            print("=" * 40)
            print("本系統需要以下額外的Python包:")
            print("• pandas - 數據處理")
            print("• openpyxl - Excel文件操作")
            print("• scikit-learn - 相似度計算")
            print("\n安裝命令:")
            print("pip install pandas openpyxl scikit-learn")
            print("\n或者安裝所有依賴:")
            print("pip install -r requirements.txt")
        
        elif choice == "5":
            print("👋 感謝使用ESG數據提取系統！")
            break
        
        else:
            print("❌ 無效選擇，請重試")

def main():
    """主函數"""
    print("🏢 ESG數據提取系統 - Excel多工作表版本 v3.0")
    print("生成包含相似關鍵字和指標統計的Excel報告")
    print("=" * 60)
    
    # 檢查命令行參數
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto":
            # 自動運行模式
            if check_environment():
                excel_path, summary, similar_groups = run_excel_extraction()
                if excel_path:
                    analyze_excel_results(excel_path)
        elif sys.argv[1] == "--check":
            # 僅檢查環境
            check_environment()
        elif sys.argv[1] == "--install":
            # 顯示安裝說明
            print("📦 請安裝以下依賴包:")
            print("pip install pandas openpyxl scikit-learn")
        else:
            print("用法:")
            print("  python excel_main.py           # 互動模式")
            print("  python excel_main.py --auto    # 自動運行")
            print("  python excel_main.py --check   # 檢查環境")
            print("  python excel_main.py --install # 安裝說明")
    else:
        # 互動模式
        interactive_menu()

if __name__ == "__main__":
    main()