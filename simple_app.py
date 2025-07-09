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
from flask import Flask, request, jsonify, render_template
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


@app.route('/')
def index():
    """主页"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>车牌识别系统</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 50px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 50px; text-align: center; margin: 20px 0; }
            .result { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }
            .error { color: red; }
            .success { color: green; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>车牌识别系统</h1>
            <div class="upload-area">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="imageFile" name="image" accept="image/*" required>
                    <br><br>
                    <button type="submit">识别车牌</button>
                </form>
            </div>
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('imageFile');
                formData.append('image', fileInput.files[0]);
                
                document.getElementById('result').innerHTML = '<p>识别中...</p>';
                
                fetch('/api/recognize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('result').innerHTML = `
                            <div class="result success">
                                <h3>识别成功！</h3>
                                <p><strong>车牌号：</strong>${data.plate_number}</p>
                                <p><strong>置信度：</strong>${(data.confidence * 100).toFixed(1)}%</p>
                            </div>
                        `;
                    } else {
                        document.getElementById('result').innerHTML = `
                            <div class="result error">
                                <h3>识别失败</h3>
                                <p>${data.error}</p>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = `
                        <div class="result error">
                            <h3>请求失败</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                });
            };
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
        
        # 执行识别
        result = recognizer.recognize(image)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'处理错误: {str(e)}'})


@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'available_engines': recognizer.engines,
        'tesseract': TESSERACT_AVAILABLE,
        'paddle': PADDLE_AVAILABLE
    })


if __name__ == '__main__':
    print("车牌识别系统启动中...")
    print(f"可用OCR引擎: {recognizer.engines}")
    print("访问 http://localhost:5000 开始使用")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
