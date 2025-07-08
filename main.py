import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pytesseract
import threading
import os
import google.generativeai as genai
import json
import base64
import time
import shutil
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("license_plate_recognition.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- 配置区 ---

# 1. Tesseract OCR 路径 (请根据你的安装位置修改)
# Windows 示例:
pytesseract.pytesseract.tesseract_cmd = r'E:\application\PDF24\tesseract\tesseract.exe'
# macOS/Linux 用户通常不需要这行，如果tesseract在系统PATH中。

# 2. Google Gemini API Key
# 强烈建议使用环境变量来保护你的API密钥
API_KEY = "AIzaSyARL4h588FeWT-eSUdQPqfZeWcmifLDjb0"
try:
    # 修复Gemini API配置问题
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    logging.info("Gemini API配置成功")
except ImportError as e:
    logging.error(f"Gemini模块导入失败: {e}. 请安装: pip install google-generativeai")
    print(f"Gemini模块导入失败: {e}")
    genai = None
except Exception as e:
    logging.error(f"Gemini API Key 配置失败: {e}")
    print(f"Gemini API Key 配置失败: {e}")
    genai = None

class HybridLPR_App:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("混合视觉模型车牌识别系统 - 天津仁爱学院")
        self.root.geometry("1200x800")
        self.root.state('zoomed')  # 最大化窗口
        
        # 初始化所有必要的属性，确保不会有未定义的属性
        self.original_image_cv = None
        self.gray_image = None
        self.binary_image = None
        self.plate_image = None
        self.video_path = None
        self.cap = None  # 视频捕获对象
        self.is_processing_video = False
        
        # 存储两种识别方法的结果
        self.tesseract_result = None
        self.gemini_result = None
        
        # 识别线程控制
        self.recognition_thread = None
        self.cancel_recognition = False
        
        # 提前初始化所有UI元素的引用
        self.video_controls_frame = None
        self.pause_button = None
        self.stop_button = None
        self.image_label = None
        self.processed_label = None
        self.tesseract_var = None
        self.gemini_var = None
        self.result_var = None
        self.engine_var = None
        self.status_var = None
        self.progress = None
        self.cancel_button = None
        
        # 首先显示登录界面
        self.setup_login_screen()
    
    def setup_login_screen(self):
        """创建登录界面"""
        self.login_frame = tk.Frame(self.root, bg="#2c3e50")
        self.login_frame.pack(expand=True, fill=tk.BOTH)
        
        # 登录标题
        title_label = tk.Label(self.login_frame, text="天津仁爱学院混合视觉车牌识别系统", font=("Arial", 24, "bold"), 
                             fg="#ecf0f1", bg="#2c3e50")
        title_label.pack(pady=(50, 30))
        
        # 登录表单
        form_frame = tk.Frame(self.login_frame, bg="#2c3e50")
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="用户名:", font=("Arial", 14), 
                fg="#ecf0f1", bg="#2c3e50").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.username_entry = tk.Entry(form_frame, font=("Arial", 14), width=20)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)
        self.username_entry.insert(0, "admin")
        
        tk.Label(form_frame, text="密码:", font=("Arial", 14), 
                fg="#ecf0f1", bg="#2c3e50").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.password_entry = tk.Entry(form_frame, show="*", font=("Arial", 14), width=20)
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)
        self.password_entry.insert(0, "admin")
        
        # 登录按钮
        login_btn = tk.Button(form_frame, text="登录", command=self.login, 
                             font=("Arial", 14, "bold"), bg="#3498db", fg="white",
                             width=15, height=1, bd=0)
        login_btn.grid(row=2, columnspan=2, pady=20)
        
        # 默认凭据提示
        cred_label = tk.Label(self.login_frame, text="默认用户名: admin, 密码: admin", 
                             font=("Arial", 12), fg="#bdc3c7", bg="#2c3e50")
        cred_label.pack(pady=10)
        
    def login(self):
        """处理登录验证"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if username == "admin" and password == "admin":
            self.login_frame.destroy()
            self.setup_ui()
        else:
            messagebox.showerror("登录失败", "用户名或密码错误")

    def setup_ui(self):
        # 主框架
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_panel = tk.Frame(main_frame, width=300, bg="#2c3e50", padx=15, pady=20)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        control_panel.pack_propagate(False)

        tk.Label(control_panel, text="控制面板", font=("微软雅黑", 18, "bold"), fg="white", bg="#2c3e50").pack(pady=(0, 20))
        
        # 文件选择区域
        file_frame = tk.Frame(control_panel, bg="#2c3e50")
        file_frame.pack(fill=tk.X, pady=5)
        
        # 图像文件选择
        tk.Button(file_frame, text="选择识别图片", command=self.select_image, 
                 font=("微软雅黑", 12), bg="#3498db", fg="white", relief=tk.FLAT, padx=10).pack(fill=tk.X, pady=5)
        
        # 视频文件选择
        tk.Button(file_frame, text="选择视频文件", command=self.select_video, 
                 font=("微软雅黑", 12), bg="#3498db", fg="white", relief=tk.FLAT, padx=10).pack(fill=tk.X, pady=5)
        
        # 状态标签
        self.status_var = tk.StringVar(value="请选择图片或视频")
        tk.Label(control_panel, textvariable=self.status_var, font=("微软雅黑", 11), fg="#bdc3c7", bg="#2c3e50", wraplength=260).pack(fill=tk.X, pady=10)

        # 进度条和取消按钮
        progress_frame = tk.Frame(control_panel, bg="#2c3e50")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.cancel_button = tk.Button(progress_frame, text="取消", command=self.cancel_recognition_process,
                                      font=("微软雅黑", 9), bg="#e74c3c", fg="white", relief=tk.FLAT)
        self.cancel_button.pack(side=tk.RIGHT)
        self.cancel_button.config(state=tk.DISABLED)

        # 图像处理按钮
        tk.Label(control_panel, text="图像处理选项", font=("微软雅黑", 14), fg="white", bg="#2c3e50").pack(pady=(10, 5))
        
        process_frame = tk.Frame(control_panel, bg="#2c3e50")
        process_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(process_frame, text="原始图像", command=self.show_original, 
                 font=("微软雅黑", 11), bg="#7f8c8d", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=3)
        tk.Button(process_frame, text="灰度图像", command=self.show_grayscale, 
                 font=("微软雅黑", 11), bg="#7f8c8d", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=3)
        tk.Button(process_frame, text="二值化图像", command=self.show_binary, 
                 font=("微软雅黑", 11), bg="#7f8c8d", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=3)
        
        # 开始识别按钮
        tk.Button(control_panel, text="开始识别", command=self.start_recognition_thread, font=("微软雅黑", 14, "bold"), bg="#27ae60", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=20)
        
        # 调试按钮 - 添加新按钮来显示调试信息
        tk.Button(control_panel, text="显示调试信息", command=self.show_debug_info, 
                 font=("微软雅黑", 11), bg="#8e44ad", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=10)

        # 视频控制按钮（初始时隐藏）
        self.video_controls_frame = tk.Frame(control_panel, bg="#2c3e50")
        self.pause_button = tk.Button(self.video_controls_frame, text="暂停", command=self.toggle_pause,
                                     font=("微软雅黑", 11), bg="#e74c3c", fg="white", relief=tk.FLAT)
        self.pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.stop_button = tk.Button(self.video_controls_frame, text="停止", command=self.stop_video,
                                    font=("微软雅黑", 11), bg="#c0392b", fg="white", relief=tk.FLAT)
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        # 视频控制按钮默认不显示，等有视频时再显示
        
        # 结果对比区 - 这部分进行了修改，显示两种识别方法的结果
        result_frame = tk.Frame(control_panel, bg="#34495e", pady=10)
        result_frame.pack(fill=tk.X, pady=20)
        tk.Label(result_frame, text="识别结果对比", font=("微软雅黑", 14, "bold"), fg="white", bg="#34495e").pack()
        
        # Tesseract结果区
        tesseract_frame = tk.Frame(result_frame, bg="#2c3e50", pady=5, padx=5)
        tesseract_frame.pack(fill=tk.X, pady=5)
        tk.Label(tesseract_frame, text="Tesseract (本地):", font=("微软雅黑", 10), fg="#bdc3c7", bg="#2c3e50").pack(anchor='w')
        self.tesseract_var = tk.StringVar(value="---")
        tk.Label(tesseract_frame, textvariable=self.tesseract_var, font=("Arial", 14, "bold"), fg="#3498db", bg="#2c3e50").pack(pady=3)
        
        # Gemini结果区
        gemini_frame = tk.Frame(result_frame, bg="#2c3e50", pady=5, padx=5)
        gemini_frame.pack(fill=tk.X, pady=5)
        tk.Label(gemini_frame, text="Gemini (云端):", font=("微软雅黑", 10), fg="#bdc3c7", bg="#2c3e50").pack(anchor='w')
        self.gemini_var = tk.StringVar(value="---")
        tk.Label(gemini_frame, textvariable=self.gemini_var, font=("Arial", 14, "bold"), fg="#e74c3c", bg="#2c3e50").pack(pady=3)
        
        # 最终结果（采用的识别结果）
        final_frame = tk.Frame(result_frame, bg="#2c3e50", pady=5, padx=5)
        final_frame.pack(fill=tk.X, pady=10)
        tk.Label(final_frame, text="最终采用:", font=("微软雅黑", 12), fg="white", bg="#2c3e50").pack(anchor='w')
        self.result_var = tk.StringVar(value="---")
        tk.Label(final_frame, textvariable=self.result_var, font=("Arial", 20, "bold"), fg="#f1c40f", bg="#2c3e50").pack(pady=3)
        self.engine_var = tk.StringVar(value="引擎: N/A")
        tk.Label(final_frame, textvariable=self.engine_var, font=("微软雅黑", 10), fg="white", bg="#2c3e50").pack()

        # 右侧图像显示区
        image_panel = tk.Frame(main_frame, bg="#ecf0f1", padx=10, pady=10)
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 上方显示原始图像
        tk.Label(image_panel, text="原始图像", font=("微软雅黑", 14)).pack()
        self.image_label = tk.Label(image_panel, bg="#ecf0f1")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 下方显示处理后的图像
        tk.Label(image_panel, text="处理结果", font=("微软雅黑", 14)).pack()
        self.processed_label = tk.Label(image_panel, bg="#ecf0f1")
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 视频处理变量
        self.is_paused = False
        
        # 存储调试信息的变量
        self.debug_logs = []

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return
        
        # 停止任何正在进行的视频处理
        self.stop_video()
        
        # 确保video_controls_frame已经初始化后再尝试使用它
        if hasattr(self, 'video_controls_frame') and self.video_controls_frame:
            self.video_controls_frame.pack_forget()
        
        try:
            # 检查文件是否存在且可读
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"找不到文件: {file_path}")
            
            # 使用PIL先打开图像，这样可以支持更多格式和路径类型
            pil_image = Image.open(file_path)
            
            # 如果是PNG图像，使用PIL处理后转换为OpenCV格式
            if file_path.lower().endswith('.png'):
                # 转换PNG为RGB模式并保存为临时JPG文件
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                
                # 将PIL图像转换为OpenCV格式
                self.original_image_cv = np.array(pil_image)
                # 由于PIL是RGB，而OpenCV是BGR，需要转换颜色通道
                self.original_image_cv = cv2.cvtColor(self.original_image_cv, cv2.COLOR_RGB2BGR)
            else:
                # 对于其他格式，尝试直接使用OpenCV读取
                self.original_image_cv = cv2.imread(file_path)
                if self.original_image_cv is None:
                    # 如果OpenCV无法读取，回退到PIL方法
                    pil_image = pil_image.convert('RGB')
                    self.original_image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 最终检查图像是否成功加载
            if self.original_image_cv is None or self.original_image_cv.size == 0:
                raise ValueError(f"无法读取图像文件: {file_path}")
            
            # 显示图像
            self.display_cv_image(self.original_image_cv, self.image_label, max_size=(800, 400))
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")
            self.result_var.set("---")
            self.engine_var.set("引擎: N/A")
            
            # 重置处理过的图像
            self.gray_image = None
            self.binary_image = None
            self.plate_image = None
            if hasattr(self, 'processed_label') and self.processed_label:
                self.processed_label.config(image='')
            
        except FileNotFoundError as e:
            messagebox.showerror("文件错误", str(e))
            self.status_var.set("文件不存在")
        except PermissionError:
            messagebox.showerror("权限错误", f"没有权限访问文件: {file_path}\n请检查文件权限或是否被OneDrive锁定")
            self.status_var.set("文件访问权限错误")
        except Exception as e:
            # 对于OneDrive文件特别处理
            if "OneDrive" in file_path:
                messagebox.showerror("OneDrive文件错误", 
                                  f"读取OneDrive文件时出错: {str(e)}\n"
                                  f"请尝试将文件下载到本地后再打开，或确保OneDrive已完全同步此文件。")
            else:
                messagebox.showerror("图像加载错误", f"加载图像时出错: {str(e)}")
            self.status_var.set("图像加载失败")

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not file_path:
            return
            
        # 关闭之前打开的视频
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.video_path = file_path
        self.status_var.set(f"已加载视频: {os.path.basename(file_path)}")
        self.result_var.set("---")
        self.engine_var.set("引擎: N/A")
        
        # 显示视频控制按钮
        self.video_controls_frame.pack(fill=tk.X, pady=10)
        
        # 打开视频文件
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return
            
        # 重置暂停状态
        self.is_paused = False
        self.pause_button.config(text="暂停")
        
        # 开始视频处理线程
        self.is_processing_video = True
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        """处理视频帧"""
        while self.is_processing_video and self.cap and self.cap.isOpened():
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    # 视频结束
                    self.is_processing_video = False
                    self.status_var.set("视频播放完毕")
                    break
                    
                # 显示当前帧
                self.original_image_cv = frame.copy()
                self.display_cv_image(frame, self.image_label, max_size=(800, 400))
                
                # 可以添加实时处理逻辑，如每隔一定帧数进行一次车牌识别
                
            time.sleep(0.033)  # ~30fps
        
        # 视频处理结束，释放资源
        if self.cap:
            self.cap.release()

    def toggle_pause(self):
        """切换视频暂停/播放状态"""
        if not self.cap or not self.is_processing_video:
            return
            
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.config(text="继续")
        else:
            self.pause_button.config(text="暂停")

    def stop_video(self):
        """停止视频处理"""
        self.is_processing_video = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.status_var.set("视频处理已停止")

    def display_cv_image(self, cv_img, label_widget, max_size):
        """显示OpenCV图像到Tkinter标签"""
        if cv_img is None:
            return
            
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail(max_size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk

    def start_recognition_thread(self):
        if self.original_image_cv is None:
            messagebox.showwarning("提示", "请先选择一张图片。")
            return
            
        # 检查图像是否有效
        if self.original_image_cv.size == 0:
            messagebox.showerror("错误", "图像数据无效，请重新选择图片。")
            return
        
        # 清除之前的调试日志
        self.debug_logs = []
        self.log_debug("开始新的识别任务")
        
        # 重置取消标志
        self.cancel_recognition = False
        
        # 启用取消按钮
        self.cancel_button.config(state=tk.NORMAL)
        
        # 启动新的识别线程
        self.recognition_thread = threading.Thread(target=self.run_hybrid_recognition, daemon=True)
        self.recognition_thread.start()
    
    def cancel_recognition_process(self):
        """取消正在进行的识别过程"""
        self.cancel_recognition = True
        self.log_debug("用户取消了识别过程")
        self.status_var.set("识别过程已取消")
        self.cancel_button.config(state=tk.DISABLED)
    
    def log_debug(self, message):
        """添加调试日志"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}"
        self.debug_logs.append(log_entry)
        logging.debug(message)
    
    def show_debug_info(self):
        """显示详细的调试信息窗口"""
        debug_window = tk.Toplevel(self.root)
        debug_window.title("调试信息")
        debug_window.geometry("800x600")
        
        # 创建文本区域用于显示调试日志
        log_frame = tk.Frame(debug_window, padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # 添加标题
        tk.Label(log_frame, text="识别过程详细日志", font=("微软雅黑", 14, "bold")).pack(pady=10)
        
        # 创建可滚动文本区域
        log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10))
        log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 插入日志内容
        if self.debug_logs:
            for log in self.debug_logs:
                log_text.insert(tk.END, log + "\n")
        else:
            log_text.insert(tk.END, "暂无识别日志。请先执行识别过程。")
        
        log_text.config(state=tk.DISABLED)  # 设为只读
        
        # 添加保存日志按钮
        save_frame = tk.Frame(debug_window)
        save_frame.pack(fill=tk.X, pady=10, padx=10)
        
        def save_logs():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"recognition_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.debug_logs))
                messagebox.showinfo("成功", f"日志已保存至: {file_path}")
                
        tk.Button(save_frame, text="保存日志", command=save_logs, 
                 font=("微软雅黑", 12), bg="#3498db", fg="white").pack(side=tk.RIGHT)
        
        # 添加关闭按钮
        tk.Button(save_frame, text="关闭", command=debug_window.destroy, 
                 font=("微软雅黑", 12), bg="#e74c3c", fg="white").pack(side=tk.LEFT)

    def recognize_with_tesseract(self, image_cv):
        """使用Tesseract进行本地OCR识别 - 增强版"""
        try:
            start_time = time.time()
            self.log_debug("开始Tesseract OCR识别")
            
            # 多种预处理方法提高识别率
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            self.log_debug("已转换为灰度图像")
            
            # 1. 直接识别原图
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
            text1 = pytesseract.image_to_string(gray, config=custom_config, lang='chi_sim+eng')
            self.log_debug(f"直接识别结果: '{text1.strip()}'")
            
            # 2. 降噪处理后识别
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            text2 = pytesseract.image_to_string(denoised, config=custom_config, lang='chi_sim+eng')
            self.log_debug(f"降噪后识别结果: '{text2.strip()}'")
            
            # 3. 二值化处理后识别
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text3 = pytesseract.image_to_string(binary, config=custom_config, lang='chi_sim+eng')
            self.log_debug(f"二值化后识别结果: '{text3.strip()}'")
            
            # 4. 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            text4 = pytesseract.image_to_string(morph, config=custom_config, lang='chi_sim+eng')
            self.log_debug(f"形态学处理后识别结果: '{text4.strip()}'")
            
            # 选择最佳结果
            results = [text1, text2, text3, text4]
            best_result = ""
            max_score = 0
            
            for text in results:
                cleaned_text = self.clean_plate_text(text)
                score = self.evaluate_plate_text(cleaned_text)
                if score > max_score:
                    max_score = score
                    best_result = cleaned_text
            
            self.log_debug(f"最佳Tesseract结果: '{best_result}' (得分: {max_score})")
            
            # 记录耗时
            elapsed_time = time.time() - start_time
            self.log_debug(f"Tesseract识别完成，耗时: {elapsed_time:.2f}秒")
            
            return best_result if max_score > 0 else ""
            
        except Exception as e:
            error_msg = f"Tesseract错误: {str(e)}"
            self.log_debug(error_msg)
            self.log_debug(traceback.format_exc())
            return ""
    
    def clean_plate_text(self, text):
        """清理车牌文本"""
        import re
        # 移除空格和换行符
        text = text.replace(' ', '').replace('\n', '').replace('\r', '')
        # 只保留中文、英文字母和数字
        text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', text)
        return text.upper()
    
    def evaluate_plate_text(self, text):
        """评估车牌文本的质量"""
        if not text:
            return 0
        
        score = 0
        # 长度评分
        if 7 <= len(text) <= 8:
            score += 50
        elif 5 <= len(text) <= 9:
            score += 30
        
        # 格式评分 - 检查是否符合中国车牌格式
        import re
        # 中国车牌格式：省份简称 + 字母 + 数字/字母组合
        plate_pattern = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{4,6}$'
        if re.match(plate_pattern, text):
            score += 40
        
        # 字符类型评分
        has_chinese = any('\u4e00' <= c <= '\u9fa5' for c in text)
        has_letter = any(c.isalpha() and c.isascii() for c in text)
        has_digit = any(c.isdigit() for c in text)
        
        if has_chinese:
            score += 20
        if has_letter:
            score += 10
        if has_digit:
            score += 10
        
        return score

    def recognize_with_gemini(self, image_cv):
        """使用Gemini Pro Vision进行云端AI识别"""
        if genai is None:
            self.log_debug("Gemini模块未正确加载")
            return {"status": "failure", "reason": "Gemini模块未正确加载"}
        
        if API_KEY == "YOUR_GOOGLE_API_KEY":
            self.log_debug("Gemini API Key未配置")
            return {"status": "failure", "reason": "Gemini API Key未配置"}
        
        try:
            start_time = time.time()
            self.log_debug("开始调用Gemini API")
            
            # 1. 初始化模型 - 使用最新的模型名称
            model = genai.GenerativeModel('gemini-1.5-flash')
            self.log_debug("已初始化Gemini模型")
            
            # 2. 图像编码
            _, buffer = cv2.imencode('.jpg', image_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            self.log_debug(f"已编码图像，大小: {len(img_base64)} 字符")
            
            # 3. 改进的Prompt
            prompt_text = """
            请仔细分析这张图片中的中国车牌号码。
            
            任务要求：
            1. 识别图片中的车牌号码
            2. 中国车牌格式：省份简称 + 字母 + 数字字母组合（如：京A12345）
            3. 请返回完整的车牌号码
            
            请严格按照以下JSON格式返回：
            {
                "status": "success",
                "plate_number": "识别的车牌号码",
                "confidence": "高/中/低"
            }
            
            如果无法识别，请返回：
            {
                "status": "failure", 
                "reason": "具体失败原因"
            }
            """
            
            self.log_debug("发送请求到Gemini API...")
            self.update_gemini_status("正在请求API...")
            
            # 4. 发送请求
            response = model.generate_content([
                prompt_text,
                {"mime_type": "image/jpeg", "data": img_base64}
            ])
            
            # 5. 处理响应
            if response.text:
                cleaned_response = response.text.strip()
                # 移除markdown格式
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
                self.log_debug(f"Gemini API响应: {cleaned_response}")
                
                result = json.loads(cleaned_response)
                
                # 记录耗时
                elapsed_time = time.time() - start_time
                self.log_debug(f"Gemini API调用完成，耗时: {elapsed_time:.2f}秒")
                
                return result
            else:
                return {"status": "failure", "reason": "API返回空响应"}
                
        except json.JSONDecodeError as e:
            error_msg = f"Gemini响应解析错误: {str(e)}"
            self.log_debug(error_msg)
            self.log_debug(f"原始响应: {response.text if 'response' in locals() else 'No response'}")
            return {"status": "failure", "reason": error_msg}
        except Exception as e:
            error_msg = f"调用Gemini API时发生错误: {str(e)}"
            self.log_debug(error_msg)
            self.log_debug(traceback.format_exc())
            return {"status": "failure", "reason": error_msg}

    def locate_plate(self, image_cv):
        """改进的车牌定位函数"""
        try:
            self.log_debug("开始车牌定位")
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # 方法1: 边缘检测 + 轮廓分析
            edges = cv2.Canny(gray, 100, 200, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 按面积排序，选择较大的轮廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:10]:  # 只检查前10个最大轮廓
                area = cv2.contourArea(contour)
                if area < 500:  # 面积太小，跳过
                    continue
                    
                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # 车牌的宽高比通常在2-6之间
                if 1.5 < aspect_ratio < 8 and w > 50 and h > 15:
                    plate_region = image_cv[y:y+h, x:x+w]
                    self.log_debug(f"检测到可能的车牌区域，宽高比: {aspect_ratio:.2f}, 尺寸: {w}x{h}")
                    
                    # 验证车牌区域的质量
                    if self.validate_plate_region(plate_region):
                        return plate_region
            
            # 方法2: 如果没有找到合适的车牌区域，尝试查找蓝色区域
            plate_region = self.find_blue_regions(image_cv)
            if plate_region is not None:
                return plate_region
            
            self.log_debug("未检测到符合条件的车牌区域")
            return None
            
        except Exception as e:
            self.log_debug(f"车牌定位失败: {str(e)}")
            return None
    
    def validate_plate_region(self, plate_region):
        """验证车牌区域的质量"""
        try:
            # 检查图像是否太小
            if plate_region.shape[0] < 15 or plate_region.shape[1] < 50:
                return False
            
            # 检查是否有足够的对比度
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            if gray.std() < 20:  # 标准差太小，可能是单色区域
                return False
            
            return True
        except:
            return False
    
    def find_blue_regions(self, image_cv):
        """查找蓝色区域（车牌背景）"""
        try:
            hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            
            # 蓝色范围
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 2 < aspect_ratio < 6:
                        plate_region = image_cv[y:y+h, x:x+w]
                        self.log_debug(f"通过蓝色检测找到车牌区域，宽高比: {aspect_ratio:.2f}")
                        return plate_region
            
            return None
        except:
            return None

    def show_original(self):
        """显示原始图像"""
        if self.original_image_cv is not None:
            self.display_cv_image(self.original_image_cv, self.processed_label, max_size=(800, 400))
            self.status_var.set("显示原始图像")
        else:
            messagebox.showwarning("提示", "请先选择一张图片")

    def show_grayscale(self):
        """显示灰度图像"""
        if self.original_image_cv is not None:
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)
            # 将灰度图转换为3通道以便显示
            gray_3channel = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
            self.display_cv_image(gray_3channel, self.processed_label, max_size=(800, 400))
            self.status_var.set("显示灰度图像")
        else:
            messagebox.showwarning("提示", "请先选择一张图片")

    def show_binary(self):
        """显示二值化图像"""
        if self.original_image_cv is not None:
            if self.binary_image is None:
                if self.gray_image is None:
                    self.gray_image = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)
                _, self.binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 将二值图转换为3通道以便显示
            binary_3channel = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
            self.display_cv_image(binary_3channel, self.processed_label, max_size=(800, 400))
            self.status_var.set("显示二值化图像")
        else:
            messagebox.showwarning("提示", "请先选择一张图片")

    def show_detailed_results(self):
        """显示详细的识别结果对话框"""
        result_window = tk.Toplevel(self.root)
        result_window.title("识别结果详情")
        result_window.geometry("600x400")
        result_window.resizable(False, False)
        
        # 主框架
        main_frame = tk.Frame(result_window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = tk.Label(main_frame, text="车牌识别结果详情", 
                              font=("微软雅黑", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 结果对比框架
        results_frame = tk.Frame(main_frame, relief=tk.RIDGE, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tesseract结果
        tesseract_frame = tk.Frame(results_frame, bg="#ecf0f1", padx=15, pady=10)
        tesseract_frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(tesseract_frame, text="Tesseract OCR (本地识别):", 
                font=("微软雅黑", 12, "bold"), bg="#ecf0f1").pack(anchor='w')
        tk.Label(tesseract_frame, text=f"结果: {self.tesseract_result or '未识别出结果'}", 
                font=("Arial", 14), bg="#ecf0f1", fg="#2c3e50").pack(anchor='w', pady=5)
        
        # Gemini结果
        gemini_frame = tk.Frame(results_frame, bg="#fdf2e9", padx=15, pady=10)
        gemini_frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(gemini_frame, text="Gemini Pro Vision (云端AI):", 
                font=("微软雅黑", 12, "bold"), bg="#fdf2e9").pack(anchor='w')
        tk.Label(gemini_frame, text=f"结果: {self.gemini_result or '未识别出结果'}", 
                font=("Arial", 14), bg="#fdf2e9", fg="#d35400").pack(anchor='w', pady=5)
        
        # 最终结果
        final_frame = tk.Frame(results_frame, bg="#e8f5e8", padx=15, pady=10)
        final_frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(final_frame, text="最终采用结果:", 
                font=("微软雅黑", 12, "bold"), bg="#e8f5e8").pack(anchor='w')
        final_result = self.result_var.get()
        tk.Label(final_frame, text=f"车牌号: {final_result}", 
                font=("Arial", 18, "bold"), bg="#e8f5e8", fg="#27ae60").pack(anchor='w', pady=5)
        
        # 按钮框架
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # 保存结果按钮
        def save_result():
            if final_result and final_result != "---" and final_result != "识别失败":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recognition_result_{timestamp}.txt"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"车牌识别结果报告\n")
                        f.write(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Tesseract结果: {self.tesseract_result or '未识别'}\n")
                        f.write(f"Gemini结果: {self.gemini_result or '未识别'}\n")
                        f.write(f"最终结果: {final_result}\n")
                        f.write(f"使用引擎: {self.engine_var.get()}\n")
                    messagebox.showinfo("保存成功", f"结果已保存到: {filename}")
                except Exception as e:
                    messagebox.showerror("保存失败", f"无法保存结果: {str(e)}")
            else:
                messagebox.showwarning("提示", "没有有效的识别结果可以保存")
        
        tk.Button(button_frame, text="保存结果", command=save_result,
                 font=("微软雅黑", 12), bg="#3498db", fg="white", 
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="关闭", command=result_window.destroy,
                 font=("微软雅黑", 12), bg="#e74c3c", fg="white", 
                 padx=20, pady=5).pack(side=tk.RIGHT, padx=5)

    def update_gemini_status(self, status):
        """更新Gemini状态显示"""
        self.gemini_var.set(status)
        self.root.update_idletasks()

    def run_hybrid_recognition(self):
        """运行混合识别流程"""
        self.progress['value'] = 0
        
        # 重置结果显示
        self.tesseract_var.set("处理中...")
        self.gemini_var.set("处理中...")
        self.result_var.set("---")
        self.engine_var.set("引擎: N/A")
        
        try:
            # --- 阶段 1: 图像预处理 ---
            self.status_var.set("阶段1: 图像预处理...")
            self.log_debug("开始图像预处理")
            self.progress['value'] = 10
            
            if self.cancel_recognition:
                self.cleanup_after_recognition()
                return
                
            # 灰度处理
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)
                self.log_debug("完成灰度转换")
            self.progress['value'] = 20
            
            if self.cancel_recognition:
                self.cleanup_after_recognition()
                return
                
            # 二值化处理
            if self.binary_image is None:
                _, self.binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.log_debug("完成二值化处理")
            self.progress['value'] = 30
            
            # --- 阶段 2: 车牌定位 ---
            self.status_var.set("阶段2: 车牌定位...")
            self.log_debug("开始车牌定位")
            self.plate_image = self.locate_plate(self.original_image_cv)
            if self.plate_image is not None:
                self.log_debug(f"车牌定位成功，尺寸: {self.plate_image.shape}")
            else:
                self.log_debug("未能定位到车牌区域")
                
            self.progress['value'] = 40
            
            if self.cancel_recognition:
                self.cleanup_after_recognition()
                return
            
            # --- 阶段 3: 开始并行识别 ---
            self.status_var.set("阶段3: 同时使用两种引擎识别...")
            self.log_debug("开始使用两种引擎识别")
            
            # 定义要识别的图像
            target_image = self.plate_image if self.plate_image is not None else self.original_image_cv
            self.log_debug(f"识别目标图像类型: {'车牌区域' if self.plate_image is not None else '原始图像'}")
            
            # 显示目标识别区域
            if self.plate_image is not None:
                self.display_cv_image(self.plate_image, self.processed_label, max_size=(800, 400))
                self.status_var.set("车牌区域提取成功，开始识别...")
            else:
                self.display_cv_image(self.original_image_cv, self.processed_label, max_size=(800, 400))
                self.status_var.set("未检测到车牌，尝试整图识别...")
            
            # --- 阶段 3a: Tesseract识别 ---
            self.progress['value'] = 50
            if self.cancel_recognition:
                self.cleanup_after_recognition()
                return
                
            self.tesseract_result = self.recognize_with_tesseract(target_image)
            self.tesseract_var.set(self.tesseract_result if self.tesseract_result else "未识别出结果")
            self.progress['value'] = 60
            
            # --- 阶段 3b: Gemini API识别 ---
            if self.cancel_recognition:
                self.cleanup_after_recognition()
                return
                
            self.progress['value'] = 70
            gemini_thread = threading.Thread(target=self.gemini_recognition_thread, args=(target_image,))
            gemini_thread.start()
            
            # 等待Gemini API线程，最多30秒
            start_time = time.time()
            while gemini_thread.is_alive():
                elapsed = time.time() - start_time
                if elapsed > 30:
                    self.log_debug("Gemini API请求超时")
                    self.gemini_var.set("请求超时")
                    break
                    
                if self.cancel_recognition:
                    self.log_debug("用户取消了识别过程")
                    self.cleanup_after_recognition()
                    return
                    
                # 更新进度条显示请求进度
                self.update_gemini_status(f"请求中... ({int(elapsed)}s)")
                time.sleep(0.5)
            
            gemini_thread.join(1)  # 再给1秒钟完成
            
            self.progress['value'] = 80
            
            # --- 阶段 4: 结果评估和选择 ---
            self.status_var.set("阶段4: 结果评估和最佳选择...")
            self.log_debug("开始评估和选择最佳结果")
            self.progress['value'] = 90
            
            # 优先使用Tesseract结果，如果符合车牌格式要求
            if self.tesseract_result and len(self.tesseract_result) >= 6:
                self.result_var.set(self.tesseract_result)
                self.engine_var.set("引擎: Tesseract OCR (本地)")
                self.status_var.set("本地识别成功!")
                self.log_debug(f"最终选择Tesseract结果: {self.tesseract_result}")
            # 次优先使用Gemini结果，如果可用
            elif self.gemini_result:
                self.result_var.set(self.gemini_result)
                self.engine_var.set("引擎: Gemini Pro Vision (云端)")
                self.status_var.set("Gemini API 识别成功!")
                self.log_debug(f"最终选择Gemini结果: {self.gemini_result}")
            # 都失败的情况
            else:
                self.result_var.set("识别失败")
                self.engine_var.set("引擎: N/A")
                self.status_var.set("所有方法均识别失败。")
                self.log_debug("所有方法均识别失败")
            
            self.progress['value'] = 100
            
            # 显示详细结果对话框
            self.show_detailed_results()
            
        except Exception as e:
            self.log_debug(f"识别过程中出现异常: {str(e)}")
            self.log_debug(traceback.format_exc())
            messagebox.showerror("错误", f"识别过程中出现异常: {str(e)}")
        finally:
            self.cleanup_after_recognition()
    
    def gemini_recognition_thread(self, image):
        """在单独的线程中运行Gemini API识别"""
        try:
            self.log_debug("Gemini识别线程开始")
            gemini_response = self.recognize_with_gemini(image)
            self.log_debug(f"Gemini返回结果: {gemini_response}")
            
            if gemini_response.get('status') == 'success':
                self.gemini_result = gemini_response.get('plate_number', 'N/A')
                self.gemini_var.set(self.gemini_result)
                self.log_debug(f"Gemini识别成功: {self.gemini_result}")
            else:
                reason = gemini_response.get('reason', '未知错误')
                self.gemini_var.set(f"失败: {reason}")
                self.gemini_result = None
                self.log_debug(f"Gemini识别失败: {reason}")
        except Exception as e:
            self.log_debug(f"Gemini线程异常: {str(e)}")
            self.log_debug(traceback.format_exc())
            self.gemini_var.set(f"错误: {str(e)}")
            self.gemini_result = None
    
    def cleanup_after_recognition(self):
        """清理识别过程资源"""
        self.cancel_button.config(state=tk.DISABLED)
        if self.cancel_recognition:
            self.progress['value'] = 0
            self.status_var.set("识别已取消")
            self.log_debug("识别过程已取消并清理")

if __name__ == "__main__":
    root = tk.Tk()
    app = HybridLPR_App(root)
    root.mainloop()