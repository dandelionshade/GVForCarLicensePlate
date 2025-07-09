#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车牌识别系统 - 简化版本
最小可行的车牌识别Web应用
"""

import os
import re
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# 尝试导入OCR库
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # 配置tesseract路径（根据实际安装路径调整）
    pytesseract.pytesseract.tesseract_cmd = r'tesseract'  # 假设已在PATH中
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
except ImportError:
    PADDLE_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test_images"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# 确保目录存在
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 支持的图像格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# 车牌相关常量
PROVINCES = ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', 
            '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕', 
            '吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼']

PLATE_PATTERN = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{5,6}$'


class SimplePlateRecognizer:
    """简化的车牌识别器"""
    
    def __init__(self):
        self.engines = []
        if TESSERACT_AVAILABLE:
            self.engines.append('tesseract')
        if PADDLE_AVAILABLE:
            self.engines.append('paddle')
    
    def detect_plate_region(self, image):
        """简单的车牌区域检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_regions = []
        for contour in contours:
            # 计算轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查宽高比和面积
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if 2.0 <= aspect_ratio <= 6.0 and 1000 <= area <= 50000:
                plate_regions.append((x, y, w, h))
        
        # 按面积排序，返回最大的几个
        plate_regions.sort(key=lambda x: x[2] * x[3], reverse=True)
        return plate_regions[:3]  # 最多返回3个候选区域
    
    def preprocess_plate(self, plate_img):
        """车牌图像预处理"""
        # 转灰度
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def recognize_with_tesseract(self, image):
        """使用Tesseract识别"""
        if not TESSERACT_AVAILABLE:
            return None
        
        try:
            # 导入pytesseract是可用的，所以可以安全地使用
            import pytesseract
            # 配置tesseract参数
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
            text = pytesseract.image_to_string(image, config=config, lang='chi_sim')
            return text.strip().replace(' ', '').replace('\n', '')
        except Exception as e:
            print(f"Tesseract识别错误: {e}")
            return None
    
    def recognize_with_paddle(self, image):
        """使用PaddleOCR识别"""
        if not PADDLE_AVAILABLE:
            return None
        
        try:
            results = paddle_ocr.ocr(image, cls=True)
            if results and results[0]:
                texts = []
                for result in results[0]:
                    if len(result) >= 2 and result[1]:
                        text = result[1][0]
                        texts.append(text)
                return ''.join(texts).replace(' ', '')
            return None
        except Exception as e:
            print(f"PaddleOCR识别错误: {e}")
            return None
    
    def validate_plate_text(self, text):
        """验证车牌文本格式"""
        if not text or len(text) < 7:
            return False
        
        # 清理文本
        text = re.sub(r'[^0-9A-Z京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]', '', text)
        
        # 检查格式
        return bool(re.match(PLATE_PATTERN, text))
    
    def recognize(self, image):
        """主要识别方法"""
        if image is None:
            return {'success': False, 'error': '图像为空'}
        
        # 检测车牌区域
        plate_regions = self.detect_plate_region(image)
        
        if not plate_regions:
            return {'success': False, 'error': '未检测到车牌区域'}
        
        results = []
        
        for x, y, w, h in plate_regions:
            # 提取车牌区域
            plate_img = image[y:y+h, x:x+w]
            
            # 预处理
            processed_img = self.preprocess_plate(plate_img)
            
            # 使用不同引擎识别
            candidates = []
            
            # Tesseract识别
            tesseract_result = self.recognize_with_tesseract(processed_img)
            if tesseract_result and self.validate_plate_text(tesseract_result):
                candidates.append(tesseract_result)
            
            # PaddleOCR识别
            paddle_result = self.recognize_with_paddle(plate_img)
            if paddle_result and self.validate_plate_text(paddle_result):
                candidates.append(paddle_result)
            
            # 选择最佳结果
            if candidates:
                best_result = max(candidates, key=len)  # 简单选择最长的结果
                results.append({
                    'text': best_result,
                    'bbox': [x, y, w, h],
                    'confidence': 0.8 if len(candidates) > 1 else 0.6
                })
        
        if results:
            # 返回置信度最高的结果
            best = max(results, key=lambda x: x['confidence'])
            return {
                'success': True,
                'plate_number': best['text'],
                'confidence': best['confidence'],
                'bbox': best['bbox'],
                'all_results': results
            }
        else:
            return {'success': False, 'error': '识别失败，未找到有效车牌号'}


# 初始化识别器
recognizer = SimplePlateRecognizer()


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_test_images():
    """获取测试图片列表"""
    images = []
    if TEST_IMAGES_DIR.exists():
        for file_path in TEST_IMAGES_DIR.glob("*"):
            if file_path.is_file() and allowed_file(file_path.name):
                # 获取文件信息
                stat = file_path.stat()
                images.append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(PROJECT_ROOT)),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
    return sorted(images, key=lambda x: x['name'])


def save_recognition_result(image_name, result):
    """保存识别结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"{timestamp}_{image_name}_result.json"
    result_path = RESULTS_DIR / result_filename
    
    # 添加时间戳和图片名到结果中
    result['timestamp'] = timestamp
    result['image_name'] = image_name
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return str(result_path)


@app.route('/')
def index():
    """主页"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>车牌识别系统</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                padding: 30px; 
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 20px;
            }
            .tabs {
                display: flex;
                background: #f8f9fa;
                border-radius: 8px;
                margin-bottom: 20px;
                overflow: hidden;
            }
            .tab {
                flex: 1;
                padding: 15px;
                text-align: center;
                cursor: pointer;
                border: none;
                background: #f8f9fa;
                transition: background-color 0.3s;
            }
            .tab.active {
                background: #007bff;
                color: white;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .upload-area { 
                border: 2px dashed #007bff; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0; 
                border-radius: 8px;
                background: #f8f9fa;
            }
            .upload-area:hover {
                background: #e9ecef;
            }
            .result { 
                margin: 20px 0; 
                padding: 20px; 
                background: #f8f9fa; 
                border-radius: 8px; 
                border-left: 4px solid #28a745;
            }
            .error { 
                color: #dc3545; 
                border-left-color: #dc3545;
            }
            .success { 
                color: #28a745; 
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .image-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                text-align: center;
                background: white;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .image-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .image-card img {
                max-width: 100%;
                height: 120px;
                object-fit: cover;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .image-card h4 {
                margin: 5px 0;
                font-size: 14px;
                color: #333;
            }
            .image-card .info {
                font-size: 12px;
                color: #666;
                margin: 2px 0;
            }
            .btn {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                transition: background-color 0.3s;
            }
            .btn:hover {
                background: #0056b3;
            }
            .btn-success {
                background: #28a745;
            }
            .btn-success:hover {
                background: #1e7e34;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚗 车牌识别系统</h1>
                <p>支持多种OCR引擎的智能车牌识别</p>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="showTab('upload')">上传识别</button>
                <button class="tab" onclick="showTab('gallery')">图片库</button>
            </div>
            
            <!-- 上传识别标签页 -->
            <div id="upload" class="tab-content active">
                <div class="upload-area">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <h3>📁 选择图片文件</h3>
                        <input type="file" id="imageFile" name="image" accept="image/*" required>
                        <br><br>
                        <button type="submit" class="btn">🔍 识别车牌</button>
                    </form>
                </div>
                <div id="uploadResult"></div>
            </div>
            
            <!-- 图片库标签页 -->
            <div id="gallery" class="tab-content">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3>📸 测试图片库</h3>
                    <button onclick="refreshGallery()" class="btn">🔄 刷新</button>
                </div>
                <div id="imageGallery" class="image-grid"></div>
                <div id="galleryResult"></div>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>正在识别中...</p>
            </div>
        </div>
        
        <script>
            // 标签页切换
            function showTab(tabName) {
                // 隐藏所有标签页内容
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // 显示选中的标签页
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
                
                // 如果是图片库标签页，加载图片
                if (tabName === 'gallery') {
                    loadGallery();
                }
            }
            
            // 上传表单提交
            document.getElementById('uploadForm').onsubmit = function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('imageFile');
                formData.append('image', fileInput.files[0]);
                
                showLoading();
                
                fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    displayResult('uploadResult', data);
                })
                .catch(error => {
                    hideLoading();
                    displayError('uploadResult', '请求失败: ' + error.message);
                });
            };
            
            // 加载图片库
            function loadGallery() {
                fetch('/api/gallery')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayGallery(data.images);
                    } else {
                        displayError('galleryResult', data.error);
                    }
                })
                .catch(error => {
                    displayError('galleryResult', '加载图片库失败: ' + error.message);
                });
            }
            
            // 显示图片库
            function displayGallery(images) {
                const gallery = document.getElementById('imageGallery');
                if (images.length === 0) {
                    gallery.innerHTML = '<p style="text-align: center; color: #666;">暂无测试图片，请将图片放入 data/test_images/ 目录</p>';
                    return;
                }
                
                gallery.innerHTML = images.map(img => `
                    <div class="image-card">
                        <img src="/api/image/${img.name}" alt="${img.name}">
                        <h4>${img.name}</h4>
                        <div class="info">大小: ${formatSize(img.size)}</div>
                        <div class="info">修改: ${img.modified}</div>
                        <button onclick="recognizeImage('${img.name}')" class="btn btn-success">识别</button>
                    </div>
                `).join('');
            }
            
            // 识别图片库中的图片
            function recognizeImage(imageName) {
                showLoading();
                
                fetch('/api/recognize_from_gallery', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image_name: imageName })
                })
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    displayResult('galleryResult', data);
                })
                .catch(error => {
                    hideLoading();
                    displayError('galleryResult', '识别失败: ' + error.message);
                });
            }
            
            // 显示结果
            function displayResult(elementId, data) {
                const resultDiv = document.getElementById(elementId);
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h3>✅ 识别成功！</h3>
                            <p><strong>车牌号：</strong>${data.plate_number}</p>
                            <p><strong>置信度：</strong>${(data.confidence * 100).toFixed(1)}%</p>
                            ${data.bbox ? `<p><strong>位置：</strong>[${data.bbox.join(', ')}]</p>` : ''}
                            ${data.result_file ? `<p><strong>结果已保存：</strong>${data.result_file}</p>` : ''}
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>❌ 识别失败</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            }
            
            // 显示错误
            function displayError(elementId, message) {
                document.getElementById(elementId).innerHTML = `
                    <div class="result error">
                        <h3>❌ 错误</h3>
                        <p>${message}</p>
                    </div>
                `;
            }
            
            // 格式化文件大小
            function formatSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            // 刷新图片库
            function refreshGallery() {
                loadGallery();
            }
            
            // 显示加载状态
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
            }
            
            // 隐藏加载状态
            function hideLoading() {
                document.getElementById('loading').style.display = 'none';
            }
        </script>
    </body>
    </html>
    '''


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API识别接口"""
    try:
        # 获取图像
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '未上传图像文件'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'})
        
        # 读取图像
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'error': '无法解码图像'})
        
        # 保存上传的图像
        save_filename = "uploaded_image"
        if file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_filename = f"{timestamp}_{filename}"
            save_path = UPLOADS_DIR / save_filename
            cv2.imwrite(str(save_path), image)
        
        # 执行识别
        result = recognizer.recognize(image)
        
        # 保存识别结果
        if result['success']:
            result_file = save_recognition_result(save_filename, result)
            result['result_file'] = str(Path(result_file).name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'处理错误: {str(e)}'})


@app.route('/api/gallery', methods=['GET'])
def api_gallery():
    """获取图片库"""
    try:
        images = get_test_images()
        return jsonify({'success': True, 'images': images})
    except Exception as e:
        return jsonify({'success': False, 'error': f'获取图片库失败: {str(e)}'})


@app.route('/api/image/<filename>')
def api_image(filename):
    """获取图片文件"""
    try:
        file_path = TEST_IMAGES_DIR / filename
        if file_path.exists() and allowed_file(filename):
            return send_file(file_path)
        else:
            return jsonify({'error': '图片不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'获取图片失败: {str(e)}'}), 500


@app.route('/api/recognize_from_gallery', methods=['POST'])
def api_recognize_from_gallery():
    """从图片库识别车牌"""
    try:
        data = request.get_json()
        if not data or 'image_name' not in data:
            return jsonify({'success': False, 'error': '缺少图片名称'})
        
        image_name = data['image_name']
        image_path = TEST_IMAGES_DIR / image_name
        
        if not image_path.exists():
            return jsonify({'success': False, 'error': '图片不存在'})
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return jsonify({'success': False, 'error': '无法读取图像文件'})
        
        # 执行识别
        result = recognizer.recognize(image)
        
        # 保存识别结果
        if result['success']:
            result_file = save_recognition_result(image_name, result)
            result['result_file'] = str(Path(result_file).name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'识别失败: {str(e)}'})


@app.route('/api/results', methods=['GET'])
def api_results():
    """获取历史识别结果"""
    try:
        results = []
        if RESULTS_DIR.exists():
            for file_path in RESULTS_DIR.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        result['file_name'] = file_path.name
                        results.append(result)
                except:
                    continue
        
        # 按时间排序
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'获取结果失败: {str(e)}'})


@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'available_engines': recognizer.engines,
        'tesseract': TESSERACT_AVAILABLE,
        'paddle': PADDLE_AVAILABLE,
        'test_images_count': len(get_test_images()),
        'directories': {
            'test_images': str(TEST_IMAGES_DIR),
            'uploads': str(UPLOADS_DIR),
            'results': str(RESULTS_DIR)
        }
    })


if __name__ == '__main__':
    print("车牌识别系统启动中...")
    print(f"可用OCR引擎: {recognizer.engines}")
    print("访问 http://localhost:5000 开始使用")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
