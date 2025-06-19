#!/usr/bin/env python3
# quick_test.py - 快速測試安裝是否成功

def test_import():
    """測試關鍵包的導入"""
    success_count = 0
    total_count = 0
    
    packages = [
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_google_genai", "langchain-google-genai"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu"),
        ("google.generativeai", "google-generativeai"),
        ("pypdf", "pypdf"),
        ("dotenv", "python-dotenv"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy")
    ]
    
    optional_packages = [
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl")
    ]
    
    print("🧪 測試核心依賴包...")
    print("=" * 50)
    
    # 測試核心包
    for import_name, package_name in packages:
        total_count += 1
        try:
            exec(f"import {import_name}")
            print(f"✅ {package_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
    
    print("\n🔧 測試可選依賴包...")
    print("=" * 50)
    
    # 測試可選包
    for import_name, package_name in optional_packages:
        try:
            exec(f"import {import_name}")
            print(f"✅ {package_name} (可選)")
        except ImportError as e:
            print(f"⚠️  {package_name} (可選): 未安裝，將使用替代方案")
    
    print("\n" + "=" * 50)
    print(f"📊 安裝結果: {success_count}/{total_count} 核心包安裝成功")
    
    if success_count == total_count:
        print("🎉 所有核心依賴安裝成功！可以開始使用系統。")
        return True
    elif success_count >= total_count - 2:
        print("⚠️  大部分依賴安裝成功，系統應該可以運行。")
        return True
    else:
        print("❌ 多個核心依賴安裝失敗，請檢查安裝。")
        return False

def test_python_version():
    """檢查Python版本"""
    import sys
    print(f"🐍 Python版本: {sys.version}")
    
    if sys.version_info >= (3, 8):
        print("✅ Python版本符合要求 (>= 3.8)")
        return True
    else:
        print("❌ Python版本過低，需要 >= 3.8")
        return False

def main():
    print("🚀 ESG RAG系統安裝檢查")
    print("=" * 50)
    
    # 檢查Python版本
    if not test_python_version():
        return
    
    print()
    
    # 檢查包安裝
    if test_import():
        print("\n✨ 系統檢查通過！")
        print("下一步：")
        print("1. 設置 Google Gemini API Key")
        print("2. 將PDF文件放入 data/ 目錄")
        print("3. 運行 python main.py")
    else:
        print("\n🔧 請根據上面的錯誤訊息重新安裝失敗的包")

if __name__ == "__main__":
    main()