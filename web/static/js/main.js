// 车牌识别系统 - 主要JavaScript功能

// 全局变量
let isUploading = false;
let currentImage = null;

// DOM 加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeComponents();
    setupEventListeners();
});

// 初始化组件
function initializeComponents() {
    // 初始化文件上传区域
    initializeFileUpload();
    
    // 初始化工具提示
    initializeTooltips();
    
    // 初始化模态框
    initializeModals();
}

// 设置事件监听器
function setupEventListeners() {
    // 文件选择事件
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // 识别按钮事件
    const recognizeBtn = document.getElementById('recognize-btn');
    if (recognizeBtn) {
        recognizeBtn.addEventListener('click', startRecognition);
    }
    
    // 清除按钮事件
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearResults);
    }
}

// 初始化文件上传区域
function initializeFileUpload() {
    const uploadArea = document.getElementById('upload-area');
    if (!uploadArea) return;
    
    // 拖拽事件
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // 点击上传
    uploadArea.addEventListener('click', function() {
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.click();
        }
    });
}

// 处理文件选择
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// 处理文件
function handleFile(file) {
    // 验证文件类型
    if (!file.type.startsWith('image/')) {
        showAlert('请选择有效的图像文件！', 'warning');
        return;
    }
    
    // 验证文件大小 (16MB)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showAlert('文件大小不能超过16MB！', 'warning');
        return;
    }
    
    // 显示预览
    displayImagePreview(file);
    
    // 启用识别按钮
    const recognizeBtn = document.getElementById('recognize-btn');
    if (recognizeBtn) {
        recognizeBtn.disabled = false;
    }
    
    // 保存当前文件
    currentImage = file;
}

// 显示图像预览
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewContainer = document.getElementById('image-preview');
        if (previewContainer) {
            previewContainer.innerHTML = `
                <div class="text-center">
                    <img src="${e.target.result}" alt="预览图像" class="result-image" style="max-height: 300px;">
                    <p class="mt-2 text-muted">文件名: ${file.name}</p>
                    <p class="text-muted">文件大小: ${formatFileSize(file.size)}</p>
                </div>
            `;
            previewContainer.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

// 开始识别
async function startRecognition() {
    if (!currentImage) {
        showAlert('请先选择图像！', 'warning');
        return;
    }
    
    if (isUploading) {
        return;
    }
    
    isUploading = true;
    
    try {
        // 更新UI
        updateRecognitionUI(true);
        
        // 获取选择的识别方法
        const method = getSelectedMethod();
        
        // 创建FormData
        const formData = new FormData();
        formData.append('image', currentImage);
        formData.append('method', method);
        
        // 发送请求
        const response = await fetch('/api/v1/recognize', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
            showAlert('识别完成！', 'success');
        } else {
            throw new Error(result.error || '识别失败');
        }
        
    } catch (error) {
        console.error('识别错误:', error);
        showAlert(`识别失败: ${error.message}`, 'danger');
    } finally {
        isUploading = false;
        updateRecognitionUI(false);
    }
}

// 获取选择的识别方法
function getSelectedMethod() {
    const methodSelect = document.getElementById('method-select');
    return methodSelect ? methodSelect.value : 'fusion';
}

// 更新识别UI状态
function updateRecognitionUI(isRecognizing) {
    const recognizeBtn = document.getElementById('recognize-btn');
    const progressBar = document.getElementById('progress-bar');
    
    if (recognizeBtn) {
        if (isRecognizing) {
            recognizeBtn.innerHTML = '<span class="loading"></span> 识别中...';
            recognizeBtn.disabled = true;
        } else {
            recognizeBtn.innerHTML = '<i class="fas fa-search"></i> 开始识别';
            recognizeBtn.disabled = !currentImage;
        }
    }
    
    if (progressBar) {
        if (isRecognizing) {
            progressBar.style.display = 'block';
            progressBar.querySelector('.progress-bar').style.width = '100%';
        } else {
            setTimeout(() => {
                progressBar.style.display = 'none';
                progressBar.querySelector('.progress-bar').style.width = '0%';
            }, 500);
        }
    }
}

// 显示识别结果
function displayResults(result) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    let resultsHTML = `
        <div class="result-container">
            <h4><i class="fas fa-check-circle text-success"></i> 识别结果</h4>
            <div class="row">
                <div class="col-md-6">
                    <h6>车牌号码:</h6>
                    <p class="fs-4 fw-bold text-primary">${result.plate_text || '未识别'}</p>
                    
                    <h6>识别方法:</h6>
                    <p>${result.method || '未知'}</p>
                    
                    <h6>置信度:</h6>
                    <div class="progress mb-2">
                        <div class="progress-bar" style="width: ${(result.confidence || 0) * 100}%">
                            ${((result.confidence || 0) * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <h6>处理时间:</h6>
                    <p>${result.processing_time || 0}秒</p>
                    
                    <h6>图像尺寸:</h6>
                    <p>${result.image_size || '未知'}</p>
                    
                    <h6>识别状态:</h6>
                    <span class="badge ${result.success ? 'bg-success' : 'bg-danger'}">
                        ${result.success ? '成功' : '失败'}
                    </span>
                </div>
            </div>
    `;
    
    // 添加详细结果
    if (result.details) {
        resultsHTML += `
            <div class="mt-3">
                <h6>详细信息:</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>引擎</th>
                                <th>结果</th>
                                <th>置信度</th>
                                <th>耗时</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        for (const [engine, data] of Object.entries(result.details)) {
            resultsHTML += `
                <tr>
                    <td>${engine}</td>
                    <td>${data.text || '识别失败'}</td>
                    <td>${data.confidence ? (data.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${data.time || 'N/A'}</td>
                </tr>
            `;
        }
        
        resultsHTML += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    resultsHTML += '</div>';
    
    resultsContainer.innerHTML = resultsHTML;
    resultsContainer.style.display = 'block';
    
    // 滚动到结果区域
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

// 清除结果
function clearResults() {
    const resultsContainer = document.getElementById('results-container');
    const previewContainer = document.getElementById('image-preview');
    const fileInput = document.getElementById('file-input');
    const recognizeBtn = document.getElementById('recognize-btn');
    
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
        resultsContainer.innerHTML = '';
    }
    
    if (previewContainer) {
        previewContainer.style.display = 'none';
        previewContainer.innerHTML = '';
    }
    
    if (fileInput) {
        fileInput.value = '';
    }
    
    if (recognizeBtn) {
        recognizeBtn.disabled = true;
    }
    
    currentImage = null;
}

// 加载统计数据
async function loadStatistics() {
    try {
        const response = await fetch('/api/v1/stats');
        const stats = await response.json();
        
        if (response.ok) {
            updateStatistics(stats);
        }
    } catch (error) {
        console.error('加载统计数据失败:', error);
    }
}

// 更新统计显示
function updateStatistics(stats) {
    const elements = {
        'total-processed': stats.total_processed || 0,
        'success-rate': stats.success_rate ? (stats.success_rate * 100).toFixed(1) + '%' : 'N/A',
        'avg-time': stats.average_time ? stats.average_time.toFixed(3) + 's' : 'N/A',
        'error-count': stats.error_count || 0
    };
    
    for (const [id, value] of Object.entries(elements)) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
}

// 显示警告信息
function showAlert(message, type = 'info') {
    // 创建警告元素
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // 添加到页面顶部
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // 3秒后自动关闭
        setTimeout(() => {
            alertDiv.remove();
        }, 3000);
    }
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 初始化工具提示
function initializeTooltips() {
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// 初始化模态框
function initializeModals() {
    // 可以在这里添加模态框相关的初始化代码
}

// 导出函数供其他脚本使用
window.PlateRecognition = {
    loadStatistics,
    showAlert,
    formatFileSize
};
