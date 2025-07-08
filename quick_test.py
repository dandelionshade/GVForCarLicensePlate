"""
快速测试脚本 - 验证车牌识别系统各组件功能
"""
import sys
import os
import time
import traceback
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """测试基础模块导入"""
    print("=== 测试基础模块导入 ===")
    
    tests = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("pytesseract", "Tesseract OCR"),
    ]
    
    results = {}
    for module, name in tests:
        try:
            __import__(module)
            results[name] = "✓ 成功"
            print(f"  {name}: ✓")
        except ImportError as e:
            results[name] = f"✗ 失败: {e}"
            print(f"  {name}: ✗ 失败")
    
    return results

def test_advanced_imports():
    """测试高级模块导入"""
    print("\\n=== 测试高级模块导入 ===")
    
    tests = [
        ("paddleocr", "PaddleOCR"),
        ("tensorflow", "TensorFlow"),
        ("torch", "PyTorch"),
        ("flask", "Flask"),
        ("prometheus_client", "Prometheus"),
    ]
    
    results = {}
    for module, name in tests:
        try:
            __import__(module)
            results[name] = "✓ 成功"
            print(f"  {name}: ✓")
        except ImportError as e:
            results[name] = f"✗ 失败 (可选): {e}"
            print(f"  {name}: ✗ 可选依赖缺失")
    
    return results

def test_project_modules():
    """测试项目模块"""
    print("\\n=== 测试项目模块 ===")
    
    modules = [
        ("config", "配置模块"),
        ("image_processor", "图像处理模块"),
        ("plate_detector", "车牌检测模块"),
        ("multi_engine_ocr", "多引擎OCR模块"),
        ("crnn_model", "CRNN模型模块"),
        ("test_framework", "测试框架模块"),
        ("api_server", "API服务模块"),
        ("optimized_system", "优化系统模块"),
    ]
    
    results = {}
    for module_name, description in modules:
        try:
            module = __import__(module_name)
            results[description] = "✓ 成功"
            print(f"  {description}: ✓")
        except ImportError as e:
            results[description] = f"✗ 失败: {e}"
            print(f"  {description}: ✗ 失败")
        except Exception as e:
            results[description] = f"⚠ 部分成功: {e}"
            print(f"  {description}: ⚠ 部分成功")
    
    return results

def test_config_validation():
    """测试配置验证"""
    print("\\n=== 测试配置验证 ===")
    
    try:
        from config import Config, ErrorCodes
        
        config = Config()
        print(f"  配置类加载: ✓")
        print(f"  车牌字符集长度: {len(config.get_character_set())}")
        print(f"  错误码定义: {len([attr for attr in dir(ErrorCodes) if not attr.startswith('_')])}")
        
        # 测试目录创建
        config.ensure_directories()
        print(f"  目录创建: ✓")
        
        return {"配置验证": "✓ 成功"}
        
    except Exception as e:
        print(f"  配置验证: ✗ 失败 - {e}")
        return {"配置验证": f"✗ 失败: {e}"}

def test_image_processing():
    """测试图像处理功能"""
    print("\\n=== 测试图像处理功能 ===")
    
    try:
        import cv2
        import numpy as np
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
        print(f"  测试图像创建: ✓ 尺寸 {test_image.shape}")
        
        # 基础图像操作
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print(f"  灰度转换: ✓")
        
        # 二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        print(f"  二值化处理: ✓")
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        print(f"  边缘检测: ✓")
        
        # 测试高级图像处理器（如果可用）
        try:
            from image_processor import AdvancedImageProcessor
            processor = AdvancedImageProcessor()
            results = processor.preprocess_image(test_image, "basic")
            print(f"  高级预处理: ✓ 生成 {len(results)} 种结果")
        except:
            print(f"  高级预处理: ⚠ 不可用")
        
        return {"图像处理": "✓ 成功"}
        
    except Exception as e:
        print(f"  图像处理: ✗ 失败 - {e}")
        return {"图像处理": f"✗ 失败: {e}"}

def test_ocr_engines():
    """测试OCR引擎"""
    print("\\n=== 测试OCR引擎 ===")
    
    results = {}
    
    # 创建简单测试图像（白底黑字）
    import cv2
    import numpy as np
    
    # 创建包含文本的图像
    img = np.ones((50, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST123", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 测试Tesseract
    try:
        import pytesseract
        text = pytesseract.image_to_string(gray, config='--psm 8').strip()
        if text:
            results["Tesseract OCR"] = f"✓ 成功: '{text}'"
            print(f"  Tesseract OCR: ✓ 识别结果: '{text}'")
        else:
            results["Tesseract OCR"] = "⚠ 无结果"
            print(f"  Tesseract OCR: ⚠ 无识别结果")
    except Exception as e:
        results["Tesseract OCR"] = f"✗ 失败: {e}"
        print(f"  Tesseract OCR: ✗ 失败")
    
    # 测试PaddleOCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(gray, cls=True)
        if result and result[0]:
            text = result[0][0][1][0]
            results["PaddleOCR"] = f"✓ 成功: '{text}'"
            print(f"  PaddleOCR: ✓ 识别结果: '{text}'")
        else:
            results["PaddleOCR"] = "⚠ 无结果"
            print(f"  PaddleOCR: ⚠ 无识别结果")
    except Exception as e:
        results["PaddleOCR"] = f"✗ 失败: {e}"
        print(f"  PaddleOCR: ✗ 失败")
    
    return results

def test_system_integration():
    """测试系统集成"""
    print("\\n=== 测试系统集成 ===")
    
    try:
        from optimized_system import OptimizedPlateRecognitionSystem
        import cv2
        import numpy as np
        
        # 初始化系统
        print("  初始化优化系统...")
        system = OptimizedPlateRecognitionSystem(enable_caching=False)
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (200, 600, 3), dtype=np.uint8)
        
        # 测试基础识别
        print("  测试基础识别...")
        start_time = time.time()
        result = system.recognize_plate(test_image, "basic")
        basic_time = time.time() - start_time
        
        print(f"    基础识别: {result.get('success', False)} (耗时: {basic_time:.3f}s)")
        
        # 测试高级识别
        print("  测试高级识别...")
        start_time = time.time()
        result = system.recognize_plate(test_image, "advanced")
        advanced_time = time.time() - start_time
        
        print(f"    高级识别: {result.get('success', False)} (耗时: {advanced_time:.3f}s)")
        
        # 测试融合识别
        print("  测试融合识别...")
        start_time = time.time()
        result = system.recognize_plate(test_image, "fusion")
        fusion_time = time.time() - start_time
        
        print(f"    融合识别: {result.get('success', False)} (耗时: {fusion_time:.3f}s)")
        
        # 获取性能报告
        performance = system.get_performance_report()
        print(f"  性能统计: 处理{performance['metrics']['total_processed']}张，平均耗时{performance['metrics']['average_time']:.3f}s")
        
        return {"系统集成": "✓ 成功"}
        
    except Exception as e:
        print(f"  系统集成: ✗ 失败 - {e}")
        traceback.print_exc()
        return {"系统集成": f"✗ 失败: {e}"}

def test_api_server():
    """测试API服务器"""
    print("\\n=== 测试API服务器 ===")
    
    try:
        from api_server import FlaskAPIServer
        
        # 创建API服务器实例
        api_server = FlaskAPIServer()
        print(f"  API服务器创建: ✓")
        
        # 检查路由注册
        app = api_server.app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        print(f"  注册路由数: {len(routes)}")
        print(f"  主要路由: {[r for r in routes if 'api' in r or r in ['/health', '/docs']]}")
        
        return {"API服务器": "✓ 成功"}
        
    except Exception as e:
        print(f"  API服务器: ✗ 失败 - {e}")
        return {"API服务器": f"✗ 失败: {e}"}

def generate_test_report(all_results):
    """生成测试报告"""
    print("\\n" + "=" * 50)
    print("测试报告汇总")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\\n【{category}】")
        if isinstance(results, dict):
            for test_name, result in results.items():
                print(f"  {test_name}: {result}")
                total_tests += 1
                if "✓" in result:
                    passed_tests += 1
        else:
            print(f"  {results}")
            total_tests += 1
            if "✓" in results:
                passed_tests += 1
    
    print(f"\\n总体情况:")
    print(f"  总测试数: {total_tests}")
    print(f"  通过测试: {passed_tests}")
    print(f"  通过率: {passed_tests/total_tests*100:.1f}%")
    
    # 系统建议
    print(f"\\n系统建议:")
    if passed_tests / total_tests >= 0.8:
        print("  ✓ 系统状态良好，可以正常使用")
    elif passed_tests / total_tests >= 0.6:
        print("  ⚠ 系统基本可用，建议安装缺失的可选依赖")
    else:
        print("  ✗ 系统存在问题，请检查依赖安装和配置")

def main():
    """主测试函数"""
    print("车牌识别系统快速测试")
    print("=" * 50)
    
    all_results = {}
    
    # 执行各项测试
    all_results["基础依赖"] = test_basic_imports()
    all_results["高级依赖"] = test_advanced_imports()
    all_results["项目模块"] = test_project_modules()
    all_results["配置验证"] = test_config_validation()
    all_results["图像处理"] = test_image_processing()
    all_results["OCR引擎"] = test_ocr_engines()
    all_results["系统集成"] = test_system_integration()
    all_results["API服务"] = test_api_server()
    
    # 生成报告
    generate_test_report(all_results)
    
    print(f"\\n测试完成！详细日志已保存至控制台输出。")

if __name__ == "__main__":
    main()
