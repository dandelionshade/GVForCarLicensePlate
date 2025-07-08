import cv2  # 导入OpenCV库，用于图像处理和视频分析
import numpy as np  # 导入NumPy库，用于进行数值计算，特别是多维数组操作
import tkinter as tk  # 导入Tkinter库，用于创建GUI应用程序
from tkinter import ttk, filedialog, messagebox, scrolledtext  # 从Tkinter导入特定的控件和对话框
from PIL import Image, ImageTk  # 导入Pillow库，用于处理图像，特别是与Tkinter的兼容性
import pytesseract  # 导入Pytesseract库，用于调用Tesseract OCR引擎进行文字识别
import threading  # 导入Threading库，用于实现多线程，防止GUI在耗时操作中冻结
import os  # 导入OS库，用于与操作系统交互，如文件路径处理
import google.generativeai as genai  # 导入Google的Generative AI库，用于调用Gemini模型
import json  # 导入JSON库，用于解析Gemini API返回的JSON格式数据
import base64  # 导入Base64库，用于将图像编码为文本字符串以发送给API
import time  # 导入Time库，用于处理时间相关的任务，如计算耗时、线程休眠
import shutil  # 导入Shutil库，提供高级文件操作功能（此项目中可能未直接使用，但保留以备扩展）
import logging  # 导入Logging库，用于记录程序运行日志，方便调试和追踪
import traceback  # 导入Traceback库，用于获取和格式化异常的详细信息
import re  # 导入re库（Regular Expression），用于进行正则表达式匹配和文本处理
from datetime import datetime  # 从datetime库导入datetime类，用于获取当前时间并格式化

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志记录的最低级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志输出的格式
    handlers=[  # 指定日志处理器列表
        logging.FileHandler("license_plate_recognition.log", encoding='utf-8'),  # 将日志写入文件，使用UTF-8编码
        logging.StreamHandler()  # 将日志输出到控制台
    ]
)

# --- 配置区 ---

# 1. Tesseract OCR 路径 (请根据你的安装位置修改)
# Windows 示例:
pytesseract.pytesseract.tesseract_cmd = r'E:\application\PDF24\tesseract\tesseract.exe'  # 设置Tesseract OCR可执行文件的路径
# macOS/Linux 用户通常不需要这行，如果tesseract在系统PATH中。

# 2. Google Gemini API Key
# 强烈建议使用环境变量来保护你的API密钥
API_KEY = "AIzaSyARL4h588FeWT-eSUdQPqfZeWcmifLDjb0"  # 设置你的Google Gemini API密钥
try:
    # 修复Gemini API配置问题
    genai.configure(api_key=API_KEY)  # 使用API密钥配置Gemini库
    logging.info("Gemini API配置成功")  # 记录配置成功的日志信息
except ImportError as e:  # 捕获导入模块失败的异常
    logging.error(f"Gemini模块导入失败: {e}. 请安装: pip install google-generativeai")  # 记录错误日志
    print(f"Gemini模块导入失败: {e}")  # 在控制台打印错误信息
    genai = None  # 将genai设置为空，表示模块不可用
except Exception as e:  # 捕获其他所有配置异常
    logging.error(f"Gemini API Key 配置失败: {e}")  # 记录API密钥配置失败的日志
    print(f"Gemini API Key 配置失败: {e}")  # 在控制台打印错误信息
    genai = None  # 将genai设置为空，表示模块不可用

class HybridLPR_App:  # 定义主应用程序类
    def __init__(self, root_window):  # 类的构造函数，初始化应用程序
        self.root = root_window  # 保存主窗口的引用
        self.root.title("混合视觉模型车牌识别系统 - 天津仁爱学院")  # 设置窗口标题
        self.root.geometry("1200x800")  # 设置窗口的初始大小
        self.root.state('zoomed')  # 默认最大化窗口

        # 初始化所有必要的属性，确保不会有未定义的属性
        self.original_image_cv = None  # 存储原始的OpenCV格式图像
        self.gray_image = None  # 存储灰度图像
        self.binary_image = None  # 存储二值化图像
        self.plate_image = None  # 存储定位到的车牌区域图像
        self.video_path = None  # 存储视频文件的路径
        self.cap = None  # 视频捕获对象，用于处理视频文件
        self.is_processing_video = False  # 标记是否正在处理视频

        # 存储两种识别方法的结果
        self.tesseract_result = None  # 存储Tesseract的识别结果
        self.gemini_result = None  # 存储Gemini的识别结果

        # 识别线程控制
        self.recognition_thread = None  # 存储识别过程的线程对象
        self.cancel_recognition = False  # 标记是否需要取消识别过程

        # 提前初始化所有UI元素的引用，避免后续引用错误
        self.video_controls_frame = None  # 视频控制按钮的容器框架
        self.pause_button = None  # 暂停/继续按钮
        self.stop_button = None  # 停止视频按钮
        self.image_label = None  # 用于显示原始图像的标签
        self.processed_label = None  # 用于显示处理后图像的标签
        self.tesseract_var = None  # Tkinter变量，用于显示Tesseract结果
        self.gemini_var = None  # Tkinter变量，用于显示Gemini结果
        self.result_var = None  # Tkinter变量，用于显示最终采纳的结果
        self.engine_var = None  # Tkinter变量，用于显示使用的识别引擎
        self.status_var = None  # Tkinter变量，用于显示当前状态信息
        self.progress = None  # 进度条控件
        self.cancel_button = None  # 取消识别按钮

        # 首先显示登录界面
        self.setup_login_screen()  # 调用方法创建并显示登录界面

    def setup_login_screen(self):  # 定义创建登录界面的方法
        """创建登录界面"""
        self.login_frame = tk.Frame(self.root, bg="#2c3e50")  # 创建一个覆盖整个窗口的框架作为登录背景
        self.login_frame.pack(expand=True, fill=tk.BOTH)  # 将框架填满整个窗口

        # 登录标题
        title_label = tk.Label(self.login_frame, text="天津仁爱学院混合视觉车牌识别系统", font=("Arial", 24, "bold"),
                             fg="#ecf0f1", bg="#2c3e50")  # 创建标题标签
        title_label.pack(pady=(50, 30))  # 将标题标签放置到框架中，并设置垂直边距

        # 登录表单
        form_frame = tk.Frame(self.login_frame, bg="#2c3e50")  # 创建用于放置登录表单的框架
        form_frame.pack(pady=20)  # 将表单框架放置到登录框架中

        tk.Label(form_frame, text="用户名:", font=("Arial", 14),
                fg="#ecf0f1", bg="#2c3e50").grid(row=0, column=0, padx=10, pady=10, sticky="e")  # 创建“用户名”标签并使用grid布局
        self.username_entry = tk.Entry(form_frame, font=("Arial", 14), width=20)  # 创建用户名输入框
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)  # 将输入框使用grid布局
        self.username_entry.insert(0, "admin")  # 插入默认用户名

        tk.Label(form_frame, text="密码:", font=("Arial", 14),
                fg="#ecf0f1", bg="#2c3e50").grid(row=1, column=0, padx=10, pady=10, sticky="e")  # 创建“密码”标签
        self.password_entry = tk.Entry(form_frame, show="*", font=("Arial", 14), width=20)  # 创建密码输入框，并设置显示为'*'
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)  # 将密码输入框使用grid布局
        self.password_entry.insert(0, "admin")  # 插入默认密码

        # 登录按钮
        login_btn = tk.Button(form_frame, text="登录", command=self.login,
                             font=("Arial", 14, "bold"), bg="#3498db", fg="white",
                             width=15, height=1, bd=0)  # 创建登录按钮
        login_btn.grid(row=2, columnspan=2, pady=20)  # 将按钮使用grid布局，跨两列

        # 默认凭据提示
        cred_label = tk.Label(self.login_frame, text="默认用户名: admin, 密码: admin",
                             font=("Arial", 12), fg="#bdc3c7", bg="#2c3e50")  # 创建提示标签
        cred_label.pack(pady=10)  # 放置提示标签

    def login(self):  # 定义处理登录逻辑的方法
        """处理登录验证"""
        username = self.username_entry.get()  # 获取用户名输入框中的内容
        password = self.password_entry.get()  # 获取密码输入框中的内容

        if username == "admin" and password == "admin":  # 检查用户名和密码是否正确
            self.login_frame.destroy()  # 如果正确，销毁登录界面
            self.setup_ui()  # 调用方法创建主应用程序界面
        else:
            messagebox.showerror("登录失败", "用户名或密码错误")  # 如果错误，显示错误提示框

    def setup_ui(self):  # 定义创建主界面的方法
        # 主框架
        main_frame = tk.Frame(self.root, bg="#f0f0f0")  # 创建主框架
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # 放置主框架

        # 左侧控制面板
        control_panel = tk.Frame(main_frame, width=300, bg="#2c3e50", padx=15, pady=20)  # 创建左侧的控制面板框架
        control_panel.pack(side=tk.LEFT, fill=tk.Y)  # 将控制面板靠左放置，并填充Y轴
        control_panel.pack_propagate(False)  # 防止控制面板因内部组件而改变大小

        tk.Label(control_panel, text="控制面板", font=("微软雅黑", 18, "bold"), fg="white", bg="#2c3e50").pack(pady=(0, 20))  # 创建控制面板标题

        # 文件选择区域
        file_frame = tk.Frame(control_panel, bg="#2c3e50")  # 创建文件选择区域的框架
        file_frame.pack(fill=tk.X, pady=5)  # 放置框架

        # 图像文件选择
        tk.Button(file_frame, text="选择识别图片", command=self.select_image,
                 font=("微软雅黑", 12), bg="#3498db", fg="white", relief=tk.FLAT, padx=10).pack(fill=tk.X, pady=5)  # 创建选择图片按钮

        # 视频文件选择
        tk.Button(file_frame, text="选择视频文件", command=self.select_video,
                 font=("微软雅黑", 12), bg="#3498db", fg="white", relief=tk.FLAT, padx=10).pack(fill=tk.X, pady=5)  # 创建选择视频按钮

        # 状态标签
        self.status_var = tk.StringVar(value="请选择图片或视频")  # 创建一个Tkinter字符串变量来动态更新状态
        tk.Label(control_panel, textvariable=self.status_var, font=("微软雅黑", 11), fg="#bdc3c7", bg="#2c3e50", wraplength=260).pack(fill=tk.X, pady=10)  # 创建状态显示标签

        # 进度条和取消按钮
        progress_frame = tk.Frame(control_panel, bg="#2c3e50")  # 创建用于放置进度条和取消按钮的框架
        progress_frame.pack(fill=tk.X, pady=5)  # 放置框架

        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")  # 创建进度条
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))  # 放置进度条

        self.cancel_button = tk.Button(progress_frame, text="取消", command=self.cancel_recognition_process,
                                      font=("微软雅黑", 9), bg="#e74c3c", fg="white", relief=tk.FLAT)  # 创建取消按钮
        self.cancel_button.pack(side=tk.RIGHT)  # 放置取消按钮
        self.cancel_button.config(state=tk.DISABLED)  # 默认禁用取消按钮

        # 图像处理按钮
        tk.Label(control_panel, text="图像处理选项", font=("微软雅黑", 14), fg="white", bg="#2c3e50").pack(pady=(10, 5))  # 创建“图像处理选项”标题

        process_frame = tk.Frame(control_panel, bg="#2c3e50")  # 创建用于放置图像处理按钮的框架
        process_frame.pack(fill=tk.X, pady=5)  # 放置框架

        tk.Button(process_frame, text="原始图像", command=self.show_original,
                 font=("微软雅黑", 11), bg="#7f8c8d", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=3)  # 创建显示原始图像按钮
        tk.Button(process_frame, text="灰度图像", command=self.show_grayscale,
                 font=("微软雅黑", 11), bg="#7f8c8d", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=3)  # 创建显示灰度图像按钮
        tk.Button(process_frame, text="二值化图像", command=self.show_binary,
                 font=("微软雅黑", 11), bg="#7f8c8d", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=3)  # 创建显示二值化图像按钮

        # 开始识别按钮
        tk.Button(control_panel, text="开始识别", command=self.start_recognition_thread, font=("微软雅黑", 14, "bold"), bg="#27ae60", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=20)  # 创建开始识别按钮

        # 调试按钮 - 添加新按钮来显示调试信息
        tk.Button(control_panel, text="显示调试信息", command=self.show_debug_info,
                 font=("微软雅黑", 11), bg="#8e44ad", fg="white", relief=tk.FLAT).pack(fill=tk.X, pady=10)  # 创建显示调试信息按钮

        # 视频控制按钮（初始时隐藏）
        self.video_controls_frame = tk.Frame(control_panel, bg="#2c3e50")  # 创建视频控制按钮的框架
        self.pause_button = tk.Button(self.video_controls_frame, text="暂停", command=self.toggle_pause,
                                     font=("微软雅黑", 11), bg="#e74c3c", fg="white", relief=tk.FLAT)  # 创建暂停/继续按钮
        self.pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)  # 放置按钮
        self.stop_button = tk.Button(self.video_controls_frame, text="停止", command=self.stop_video,
                                    font=("微软雅黑", 11), bg="#c0392b", fg="white", relief=tk.FLAT)  # 创建停止按钮
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)  # 放置按钮
        # 视频控制按钮默认不显示，等有视频时再显示

        # 结果对比区 - 这部分进行了修改，显示两种识别方法的结果
        result_frame = tk.Frame(control_panel, bg="#34495e", pady=10)  # 创建结果对比区的框架
        result_frame.pack(fill=tk.X, pady=20)  # 放置框架
        tk.Label(result_frame, text="识别结果对比", font=("微软雅黑", 14, "bold"), fg="white", bg="#34495e").pack()  # 创建标题

        # Tesseract结果区
        tesseract_frame = tk.Frame(result_frame, bg="#2c3e50", pady=5, padx=5)  # 创建Tesseract结果的框架
        tesseract_frame.pack(fill=tk.X, pady=5)  # 放置框架
        tk.Label(tesseract_frame, text="Tesseract (本地):", font=("微软雅黑", 10), fg="#bdc3c7", bg="#2c3e50").pack(anchor='w')  # 创建标签
        self.tesseract_var = tk.StringVar(value="---")  # 创建Tkinter变量用于显示Tesseract结果
        tk.Label(tesseract_frame, textvariable=self.tesseract_var, font=("Arial", 14, "bold"), fg="#3498db", bg="#2c3e50").pack(pady=3)  # 创建显示结果的标签

        # Gemini结果区
        gemini_frame = tk.Frame(result_frame, bg="#2c3e50", pady=5, padx=5)  # 创建Gemini结果的框架
        gemini_frame.pack(fill=tk.X, pady=5)  # 放置框架
        tk.Label(gemini_frame, text="Gemini (云端):", font=("微软雅黑", 10), fg="#bdc3c7", bg="#2c3e50").pack(anchor='w')  # 创建标签
        self.gemini_var = tk.StringVar(value="---")  # 创建Tkinter变量用于显示Gemini结果
        tk.Label(gemini_frame, textvariable=self.gemini_var, font=("Arial", 14, "bold"), fg="#e74c3c", bg="#2c3e50").pack(pady=3)  # 创建显示结果的标签

        # 最终结果（采用的识别结果）
        final_frame = tk.Frame(result_frame, bg="#2c3e50", pady=5, padx=5)  # 创建最终结果的框架
        final_frame.pack(fill=tk.X, pady=10)  # 放置框架
        tk.Label(final_frame, text="最终采用:", font=("微软雅黑", 12), fg="white", bg="#2c3e50").pack(anchor='w')  # 创建标签
        self.result_var = tk.StringVar(value="---")  # 创建Tkinter变量用于显示最终结果
        tk.Label(final_frame, textvariable=self.result_var, font=("Arial", 20, "bold"), fg="#f1c40f", bg="#2c3e50").pack(pady=3)  # 创建显示结果的标签
        self.engine_var = tk.StringVar(value="引擎: N/A")  # 创建Tkinter变量用于显示使用的引擎
        tk.Label(final_frame, textvariable=self.engine_var, font=("微软雅黑", 10), fg="white", bg="#2c3e50").pack()  # 创建显示引擎的标签

        # 右侧图像显示区
        image_panel = tk.Frame(main_frame, bg="#ecf0f1", padx=10, pady=10)  # 创建右侧图像显示区的框架
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # 靠右放置并填满剩余空间

        # 上方显示原始图像
        tk.Label(image_panel, text="原始图像", font=("微软雅黑", 14)).pack()  # 创建标题
        self.image_label = tk.Label(image_panel, bg="#ecf0f1")  # 创建用于显示原始图像的标签
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # 放置标签

        # 下方显示处理后的图像
        tk.Label(image_panel, text="处理结果", font=("微软雅黑", 14)).pack()  # 创建标题
        self.processed_label = tk.Label(image_panel, bg="#ecf0f1")  # 创建用于显示处理后图像的标签
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # 放置标签

        # 视频处理变量
        self.is_paused = False  # 标记视频是否处于暂停状态

        # 存储调试信息的变量
        self.debug_logs = []  # 初始化一个列表来存储调试日志

    def select_image(self):  # 定义选择和加载图片的方法
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])  # 打开文件选择对话框
        if not file_path:  # 如果用户取消选择，则直接返回
            return

        # 停止任何正在进行的视频处理
        self.stop_video()  # 调用停止视频的方法

        # 确保video_controls_frame已经初始化后再尝试使用它
        if hasattr(self, 'video_controls_frame') and self.video_controls_frame:  # 检查视频控制框架是否存在
            self.video_controls_frame.pack_forget()  # 如果存在，则隐藏它

        try:
            # 检查文件是否存在且可读
            if not os.path.isfile(file_path):  # 检查路径是否为文件
                raise FileNotFoundError(f"找不到文件: {file_path}")  # 如果不是，则抛出异常

            # 使用PIL先打开图像，这样可以支持更多格式和路径类型
            pil_image = Image.open(file_path)  # 使用Pillow库打开图像

            # 如果是PNG图像，使用PIL处理后转换为OpenCV格式
            if file_path.lower().endswith('.png'):  # 检查文件是否为PNG格式
                # 转换PNG为RGB模式并保存为临时JPG文件
                if pil_image.mode == 'RGBA':  # 如果PNG有Alpha通道
                    pil_image = pil_image.convert('RGB')  # 转换为RGB模式

                # 将PIL图像转换为OpenCV格式
                self.original_image_cv = np.array(pil_image)  # 将Pillow图像转换为NumPy数组
                # 由于PIL是RGB，而OpenCV是BGR，需要转换颜色通道
                self.original_image_cv = cv2.cvtColor(self.original_image_cv, cv2.COLOR_RGB2BGR)  # 转换颜色空间
            else:
                # 对于其他格式，尝试直接使用OpenCV读取
                self.original_image_cv = cv2.imread(file_path)  # 使用OpenCV读取图像
                if self.original_image_cv is None:  # 如果OpenCV读取失败
                    # 如果OpenCV无法读取，回退到PIL方法
                    pil_image = pil_image.convert('RGB')  # 确保Pillow图像是RGB模式
                    self.original_image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # 再次尝试转换

            # 最终检查图像是否成功加载
            if self.original_image_cv is None or self.original_image_cv.size == 0:  # 检查图像数据是否有效
                raise ValueError(f"无法读取图像文件: {file_path}")  # 如果无效，则抛出异常

            # 显示图像
            self.display_cv_image(self.original_image_cv, self.image_label, max_size=(800, 400))  # 调用方法显示图像
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")  # 更新状态栏信息
            self.result_var.set("---")  # 重置结果显示
            self.engine_var.set("引擎: N/A")  # 重置引擎显示

            # 重置处理过的图像
            self.gray_image = None  # 清空灰度图像缓存
            self.binary_image = None  # 清空二值化图像缓存
            self.plate_image = None  # 清空车牌图像缓存
            if hasattr(self, 'processed_label') and self.processed_label:  # 检查处理后图像标签是否存在
                self.processed_label.config(image='')  # 清空其内容

        except FileNotFoundError as e:  # 捕获文件未找到异常
            messagebox.showerror("文件错误", str(e))  # 显示错误提示框
            self.status_var.set("文件不存在")  # 更新状态
        except PermissionError:  # 捕获权限错误
            messagebox.showerror("权限错误", f"没有权限访问文件: {file_path}\n请检查文件权限或是否被OneDrive锁定")  # 显示错误提示框
            self.status_var.set("文件访问权限错误")  # 更新状态
        except Exception as e:  # 捕获其他所有异常
            # 对于OneDrive文件特别处理
            if "OneDrive" in file_path:  # 如果文件路径包含"OneDrive"
                messagebox.showerror("OneDrive文件错误",
                                  f"读取OneDrive文件时出错: {str(e)}\n"
                                  f"请尝试将文件下载到本地后再打开，或确保OneDrive已完全同步此文件。")  # 显示特定于OneDrive的错误
            else:
                messagebox.showerror("图像加载错误", f"加载图像时出错: {str(e)}")  # 显示通用加载错误
            self.status_var.set("图像加载失败")  # 更新状态

    def select_video(self):  # 定义选择和加载视频的方法
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])  # 打开文件选择对话框
        if not file_path:  # 如果用户取消选择，则返回
            return

        # 关闭之前打开的视频
        if self.cap and self.cap.isOpened():  # 检查是否有已打开的视频
            self.cap.release()  # 释放视频资源

        self.video_path = file_path  # 保存视频路径
        self.status_var.set(f"已加载视频: {os.path.basename(file_path)}")  # 更新状态栏
        self.result_var.set("---")  # 重置结果显示
        self.engine_var.set("引擎: N/A")  # 重置引擎显示

        # 显示视频控制按钮
        self.video_controls_frame.pack(fill=tk.X, pady=10)  # 显示视频控制框架

        # 打开视频文件
        self.cap = cv2.VideoCapture(file_path)  # 使用OpenCV打开视频文件
        if not self.cap.isOpened():  # 检查视频是否成功打开
            messagebox.showerror("错误", "无法打开视频文件")  # 如果失败，显示错误
            return

        # 重置暂停状态
        self.is_paused = False  # 将暂停标记设为False
        self.pause_button.config(text="暂停")  # 将按钮文本设为“暂停”

        # 开始视频处理线程
        self.is_processing_video = True  # 设置视频处理标记为True
        threading.Thread(target=self.process_video, daemon=True).start()  # 在新线程中开始处理视频

    def process_video(self):  # 定义处理视频帧的方法
        """处理视频帧"""
        while self.is_processing_video and self.cap and self.cap.isOpened():  # 循环直到视频结束或被停止
            if not self.is_paused:  # 如果没有暂停
                ret, frame = self.cap.read()  # 读取一帧
                if not ret:  # 如果读取失败（视频结束）
                    # 视频结束
                    self.is_processing_video = False  # 更新处理标记
                    self.status_var.set("视频播放完毕")  # 更新状态
                    break  # 退出循环

                # 显示当前帧
                self.original_image_cv = frame.copy()  # 复制当前帧以备后续识别
                self.display_cv_image(frame, self.image_label, max_size=(800, 400))  # 显示当前帧

                # 可以添加实时处理逻辑，如每隔一定帧数进行一次车牌识别

            time.sleep(0.033)  # 等待约33毫秒，以模拟约30fps的播放速度

        # 视频处理结束，释放资源
        if self.cap:  # 检查视频捕获对象是否存在
            self.cap.release()  # 释放资源

    def toggle_pause(self):  # 定义切换视频暂停/播放状态的方法
        """切换视频暂停/播放状态"""
        if not self.cap or not self.is_processing_video:  # 检查视频是否正在处理
            return

        self.is_paused = not self.is_paused  # 切换暂停状态
        if self.is_paused:  # 如果现在是暂停状态
            self.pause_button.config(text="继续")  # 按钮文本改为“继续”
        else:  # 如果现在是播放状态
            self.pause_button.config(text="暂停")  # 按钮文本改为“暂停”

    def stop_video(self):  # 定义停止视频处理的方法
        """停止视频处理"""
        self.is_processing_video = False  # 设置处理标记为False
        if self.cap and self.cap.isOpened():  # 检查视频捕获对象
            self.cap.release()  # 释放资源
            self.cap = None  # 将对象设为None
        self.status_var.set("视频处理已停止")  # 更新状态

    def display_cv_image(self, cv_img, label_widget, max_size):  # 定义在Tkinter标签中显示OpenCV图像的方法
        """显示OpenCV图像到Tkinter标签"""
        if cv_img is None:  # 检查图像数据是否有效
            return

        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # 将OpenCV的BGR格式转换为RGB格式
        img_pil = Image.fromarray(img_rgb)  # 将NumPy数组转换为Pillow图像
        img_pil.thumbnail(max_size, Image.Resampling.LANCZOS)  # 按比例缩放图像以适应窗口
        img_tk = ImageTk.PhotoImage(img_pil)  # 将Pillow图像转换为Tkinter兼容的格式
        label_widget.config(image=img_tk)  # 在标签控件中设置图像
        label_widget.image = img_tk  # 保持对图像对象的引用，防止被垃圾回收

    def start_recognition_thread(self):  # 定义启动识别线程的方法
        if self.original_image_cv is None:  # 检查是否已加载图像
            messagebox.showwarning("提示", "请先选择一张图片。")  # 如果没有，则提示用户
            return

        # 检查图像是否有效
        if self.original_image_cv.size == 0:  # 检查图像数据是否为空
            messagebox.showerror("错误", "图像数据无效，请重新选择图片。")  # 如果无效，则显示错误
            return

        # 清除之前的调试日志
        self.debug_logs = []  # 重置调试日志列表
        self.log_debug("开始新的识别任务")  # 记录日志

        # 重置取消标志
        self.cancel_recognition = False  # 将取消标记设为False

        # 启用取消按钮
        self.cancel_button.config(state=tk.NORMAL)  # 启用取消按钮

        # 启动新的识别线程
        self.recognition_thread = threading.Thread(target=self.run_hybrid_recognition, daemon=True)  # 创建识别线程
        self.recognition_thread.start()  # 启动线程

    def cancel_recognition_process(self):  # 定义取消识别过程的方法
        """取消正在进行的识别过程"""
        self.cancel_recognition = True  # 设置取消标记为True
        self.log_debug("用户取消了识别过程")  # 记录日志
        self.status_var.set("识别过程已取消")  # 更新状态
        self.cancel_button.config(state=tk.DISABLED)  # 禁用取消按钮

    def log_debug(self, message):  # 定义记录调试日志的方法
        """添加调试日志"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 获取带毫秒的时间戳
        log_entry = f"[{timestamp}] {message}"  # 格式化日志条目
        self.debug_logs.append(log_entry)  # 将日志条目添加到列表中
        logging.debug(message)  # 使用logging模块记录日志

    def show_debug_info(self):  # 定义显示调试信息窗口的方法
        """显示详细的调试信息窗口"""
        debug_window = tk.Toplevel(self.root)  # 创建一个新的顶级窗口
        debug_window.title("调试信息")  # 设置窗口标题
        debug_window.geometry("800x600")  # 设置窗口大小

        # 创建文本区域用于显示调试日志
        log_frame = tk.Frame(debug_window, padx=10, pady=10)  # 创建框架
        log_frame.pack(fill=tk.BOTH, expand=True)  # 放置框架

        # 添加标题
        tk.Label(log_frame, text="识别过程详细日志", font=("微软雅黑", 14, "bold")).pack(pady=10)  # 创建标题

        # 创建可滚动文本区域
        log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10))  # 创建带滚动条的文本框
        log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # 放置文本框

        # 插入日志内容
        if self.debug_logs:  # 检查是否有日志
            for log in self.debug_logs:  # 遍历所有日志
                log_text.insert(tk.END, log + "\n")  # 插入到文本框
        else:
            log_text.insert(tk.END, "暂无识别日志。请先执行识别过程。")  # 显示提示信息

        log_text.config(state=tk.DISABLED)  # 将文本框设为只读

        # 添加保存日志按钮
        save_frame = tk.Frame(debug_window)  # 创建按钮框架
        save_frame.pack(fill=tk.X, pady=10, padx=10)  # 放置框架

        def save_logs():  # 定义保存日志的内部函数
            file_path = filedialog.asksaveasfilename(  # 打开文件保存对话框
                defaultextension=".txt",  # 默认扩展名
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],  # 文件类型过滤器
                initialfile=f"recognition_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"  # 默认文件名
            )
            if file_path:  # 如果用户选择了路径
                with open(file_path, 'w', encoding='utf-8') as f:  # 打开文件
                    f.write("\n".join(self.debug_logs))  # 写入日志
                messagebox.showinfo("成功", f"日志已保存至: {file_path}")  # 显示成功信息

        tk.Button(save_frame, text="保存日志", command=save_logs,
                 font=("微软雅黑", 12), bg="#3498db", fg="white").pack(side=tk.RIGHT)  # 创建并放置保存按钮

        # 添加关闭按钮
        tk.Button(save_frame, text="关闭", command=debug_window.destroy,
                 font=("微软雅黑", 12), bg="#e74c3c", fg="white").pack(side=tk.LEFT)  # 创建并放置关闭按钮

    def recognize_with_tesseract(self, image_cv):  # 定义使用Tesseract进行识别的方法
        """使用Tesseract进行本地OCR识别 - 增强版"""
        try:
            start_time = time.time()  # 记录开始时间
            self.log_debug("开始Tesseract OCR识别")  # 记录日志

            # 多种预处理方法提高识别率
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
            self.log_debug("已转换为灰度图像")  # 记录日志

            # 1. 直接识别原图
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'  # Tesseract配置参数
            text1 = pytesseract.image_to_string(gray, config=custom_config, lang='chi_sim+eng')  # 进行识别
            self.log_debug(f"直接识别结果: '{text1.strip()}'")  # 记录结果

            # 2. 降噪处理后识别
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)  # 图像降噪
            text2 = pytesseract.image_to_string(denoised, config=custom_config, lang='chi_sim+eng')  # 再次识别
            self.log_debug(f"降噪后识别结果: '{text2.strip()}'")  # 记录结果

            # 3. 二值化处理后识别
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 图像二值化
            text3 = pytesseract.image_to_string(binary, config=custom_config, lang='chi_sim+eng')  # 再次识别
            self.log_debug(f"二值化后识别结果: '{text3.strip()}'")  # 记录结果

            # 4. 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义形态学操作的核
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 执行闭运算
            text4 = pytesseract.image_to_string(morph, config=custom_config, lang='chi_sim+eng')  # 再次识别
            self.log_debug(f"形态学处理后识别结果: '{text4.strip()}'")  # 记录结果

            # 选择最佳结果
            results = [text1, text2, text3, text4]  # 将所有结果放入列表
            best_result = ""  # 初始化最佳结果
            max_score = 0  # 初始化最高分

            for text in results:  # 遍历所有结果
                cleaned_text = self.clean_plate_text(text)  # 清理文本
                score = self.evaluate_plate_text(cleaned_text)  # 评估文本质量
                if score > max_score:  # 如果当前得分更高
                    max_score = score  # 更新最高分
                    best_result = cleaned_text  # 更新最佳结果

            self.log_debug(f"最佳Tesseract结果: '{best_result}' (得分: {max_score})")  # 记录最终选择

            # 记录耗时
            elapsed_time = time.time() - start_time  # 计算耗时
            self.log_debug(f"Tesseract识别完成，耗时: {elapsed_time:.2f}秒")  # 记录耗时

            return best_result if max_score > 0 else ""  # 返回最佳结果，如果没有好的结果则返回空字符串

        except Exception as e:  # 捕获异常
            error_msg = f"Tesseract错误: {str(e)}"  # 格式化错误信息
            self.log_debug(error_msg)  # 记录错误
            self.log_debug(traceback.format_exc())  # 记录详细的堆栈跟踪
            return ""  # 返回空字符串

    def clean_plate_text(self, text):  # 定义清理车牌文本的方法
        """清理车牌文本"""
        # 移除空格和换行符
        text = text.replace(' ', '').replace('\n', '').replace('\r', '')  # 替换掉空白字符
        # 只保留中文、英文字母和数字
        text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', text)  # 使用正则表达式移除无效字符
        return text.upper()  # 返回大写形式的文本

    def evaluate_plate_text(self, text):  # 定义评估车牌文本质量的方法
        """评估车牌文本的质量"""
        if not text:  # 如果文本为空
            return 0  # 返回0分

        score = 0  # 初始化分数
        # 长度评分
        if 7 <= len(text) <= 8:  # 如果长度在7到8之间（标准车牌长度）
            score += 50  # 加50分
        elif 5 <= len(text) <= 9:  # 如果长度在合理范围内
            score += 30  # 加30分

        # 格式评分 - 检查是否符合中国车牌格式
        import re  # 再次导入re（虽然已在顶部导入，但在此处可作为提醒）
        # 中国车牌格式：省份简称 + 字母 + 数字/字母组合
        plate_pattern = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][0-9A-Z]{4,6}$'  # 定义车牌的正则表达式
        if re.match(plate_pattern, text):  # 如果文本匹配该模式
            score += 40  # 加40分

        # 字符类型评分
        has_chinese = any('\u4e00' <= c <= '\u9fa5' for c in text)  # 检查是否包含汉字
        has_letter = any(c.isalpha() and c.isascii() for c in text)  # 检查是否包含英文字母
        has_digit = any(c.isdigit() for c in text)  # 检查是否包含数字

        if has_chinese:  # 如果有汉字
            score += 20  # 加20分
        if has_letter:  # 如果有字母
            score += 10  # 加10分
        if has_digit:  # 如果有数字
            score += 10  # 加10分

        return score  # 返回总分

    def recognize_with_gemini(self, image_cv):  # 定义使用Gemini进行识别的方法
        """使用Gemini Pro Vision进行云端AI识别"""
        if genai is None:  # 检查Gemini模块是否加载成功
            self.log_debug("Gemini模块未正确加载")  # 记录日志
            return {"status": "failure", "reason": "Gemini模块未正确加载"}  # 返回失败信息

        if API_KEY == "YOUR_GOOGLE_API_KEY":  # 检查API密钥是否已配置
            self.log_debug("Gemini API Key未配置")  # 记录日志
            return {"status": "failure", "reason": "Gemini API Key未配置"}  # 返回失败信息

        try:
            start_time = time.time()  # 记录开始时间
            self.log_debug("开始调用Gemini API")  # 记录日志

            # 1. 初始化模型 - 使用最新的模型名称
            model = genai.GenerativeModel('gemini-1.5-flash')  # 初始化Gemini模型
            self.log_debug("已初始化Gemini模型")  # 记录日志

            # 2. 图像编码
            _, buffer = cv2.imencode('.jpg', image_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 将图像编码为JPEG格式
            img_base64 = base64.b64encode(buffer).decode('utf-8')  # 将JPEG数据进行Base64编码
            self.log_debug(f"已编码图像，大小: {len(img_base64)} 字符")  # 记录日志

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
            """  # 定义给Gemini的提示词

            self.log_debug("发送请求到Gemini API...")  # 记录日志
            self.update_gemini_status("正在请求API...")  # 更新UI状态

            # 4. 发送请求
            response = model.generate_content([  # 调用模型生成内容
                prompt_text,  # 提示文本
                {"mime_type": "image/jpeg", "data": img_base64}  # 图像数据
            ])

            # 5. 处理响应
            if response.text:  # 检查响应是否有文本内容
                cleaned_response = response.text.strip()  # 去除首尾空白
                # 移除markdown格式
                cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()  # 清理响应中的Markdown标记
                self.log_debug(f"Gemini API响应: {cleaned_response}")  # 记录日志

                result = json.loads(cleaned_response)  # 解析JSON字符串

                # 记录耗时
                elapsed_time = time.time() - start_time  # 计算耗时
                self.log_debug(f"Gemini API调用完成，耗时: {elapsed_time:.2f}秒")  # 记录日志

                return result  # 返回解析后的结果
            else:
                return {"status": "failure", "reason": "API返回空响应"}  # 如果响应为空，返回失败信息

        except json.JSONDecodeError as e:  # 捕获JSON解析错误
            error_msg = f"Gemini响应解析错误: {str(e)}"  # 格式化错误信息
            self.log_debug(error_msg)  # 记录日志
            self.log_debug(f"原始响应: {response.text if 'response' in locals() else 'No response'}")  # 记录原始响应内容
            return {"status": "failure", "reason": error_msg}  # 返回失败信息
        except Exception as e:  # 捕获其他所有异常
            error_msg = f"调用Gemini API时发生错误: {str(e)}"  # 格式化错误信息
            self.log_debug(error_msg)  # 记录日志
            self.log_debug(traceback.format_exc())  # 记录详细堆栈跟踪
            return {"status": "failure", "reason": error_msg}  # 返回失败信息

    def locate_plate(self, image_cv):  # 定义车牌定位的方法
        """改进的车牌定位函数"""
        try:
            self.log_debug("开始车牌定位")  # 记录日志
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

            # 方法1: 边缘检测 + 轮廓分析
            edges = cv2.Canny(gray, 100, 200, apertureSize=3)  # Canny边缘检测
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

            # 按面积排序，选择较大的轮廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 按面积降序排序轮廓

            for contour in contours[:10]:  # 只检查前10个最大轮廓
                area = cv2.contourArea(contour)  # 计算轮廓面积
                if area < 500:  # 如果面积太小，则跳过
                    continue

                # 获取轮廓的边界矩形
                x, y, w, h = cv2.boundingRect(contour)  # 获取边界矩形
                aspect_ratio = w / h  # 计算宽高比

                # 车牌的宽高比通常在2-6之间
                if 1.5 < aspect_ratio < 8 and w > 50 and h > 15:  # 根据宽高比和尺寸筛选
                    plate_region = image_cv[y:y+h, x:x+w]  # 裁剪出可能的车牌区域
                    self.log_debug(f"检测到可能的车牌区域，宽高比: {aspect_ratio:.2f}, 尺寸: {w}x{h}")  # 记录日志

                    # 验证车牌区域的质量
                    if self.validate_plate_region(plate_region):  # 调用验证函数
                        return plate_region  # 如果验证通过，返回该区域

            # 方法2: 如果没有找到合适的车牌区域，尝试查找蓝色区域
            plate_region = self.find_blue_regions(image_cv)  # 调用查找蓝色区域的方法
            if plate_region is not None:  # 如果找到了
                return plate_region  # 返回该区域

            self.log_debug("未检测到符合条件的车牌区域")  # 记录日志
            return None  # 如果所有方法都失败，返回None

        except Exception as e:  # 捕获异常
            self.log_debug(f"车牌定位失败: {str(e)}")  # 记录日志
            return None  # 返回None

    def validate_plate_region(self, plate_region):  # 定义验证车牌区域质量的方法
        """验证车牌区域的质量"""
        try:
            # 检查图像是否太小
            if plate_region.shape[0] < 15 or plate_region.shape[1] < 50:  # 检查尺寸是否过小
                return False  # 如果是，则验证失败

            # 检查是否有足够的对比度
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
            if gray.std() < 20:  # 计算灰度图的标准差，如果太小说明对比度低
                return False  # 验证失败

            return True  # 如果通过所有检查，则验证成功
        except:
            return False  # 如果发生任何错误，则验证失败

    def find_blue_regions(self, image_cv):  # 定义通过颜色查找蓝色区域的方法
        """查找蓝色区域（车牌背景）"""
        try:
            hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)  # 转换到HSV颜色空间

            # 蓝色范围
            lower_blue = np.array([100, 50, 50])  # 定义蓝色的下限
            upper_blue = np.array([130, 255, 255])  # 定义蓝色的上限

            mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 创建蓝色的掩码
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

            for contour in contours:  # 遍历所有轮廓
                area = cv2.contourArea(contour)  # 计算面积
                if area > 1000:  # 过滤掉小面积区域
                    x, y, w, h = cv2.boundingRect(contour)  # 获取边界矩形
                    aspect_ratio = w / h  # 计算宽高比
                    if 2 < aspect_ratio < 6:  # 检查宽高比是否符合车牌特征
                        plate_region = image_cv[y:y+h, x:x+w]  # 裁剪出区域
                        self.log_debug(f"通过蓝色检测找到车牌区域，宽高比: {aspect_ratio:.2f}")  # 记录日志
                        return plate_region  # 返回区域

            return None  # 如果没有找到，返回None
        except:
            return None  # 如果发生错误，返回None

    def show_original(self):  # 定义显示原始图像的方法
        """显示原始图像"""
        if self.original_image_cv is not None:  # 检查原始图像是否存在
            self.display_cv_image(self.original_image_cv, self.processed_label, max_size=(800, 400))  # 在处理结果区显示原始图像
            self.status_var.set("显示原始图像")  # 更新状态
        else:
            messagebox.showwarning("提示", "请先选择一张图片")  # 提示用户

    def show_grayscale(self):  # 定义显示灰度图像的方法
        """显示灰度图像"""
        if self.original_image_cv is not None:  # 检查原始图像是否存在
            if self.gray_image is None:  # 如果灰度图未被缓存
                self.gray_image = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)  # 生成灰度图
            # 将灰度图转换为3通道以便显示
            gray_3channel = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)  # 转换为3通道图像
            self.display_cv_image(gray_3channel, self.processed_label, max_size=(800, 400))  # 显示图像
            self.status_var.set("显示灰度图像")  # 更新状态
        else:
            messagebox.showwarning("提示", "请先选择一张图片")  # 提示用户

    def show_binary(self):  # 定义显示二值化图像的方法
        """显示二值化图像"""
        if self.original_image_cv is not None:  # 检查原始图像是否存在
            if self.binary_image is None:  # 如果二值化图像未被缓存
                if self.gray_image is None:  # 如果灰度图也未被缓存
                    self.gray_image = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)  # 先生成灰度图
                _, self.binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 生成二值化图像
            # 将二值图转换为3通道以便显示
            binary_3channel = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)  # 转换为3通道图像
            self.display_cv_image(binary_3channel, self.processed_label, max_size=(800, 400))  # 显示图像
            self.status_var.set("显示二值化图像")  # 更新状态
        else:
            messagebox.showwarning("提示", "请先选择一张图片")  # 提示用户

    def show_detailed_results(self):  # 定义显示详细结果对话框的方法
        """显示详细的识别结果对话框"""
        result_window = tk.Toplevel(self.root)  # 创建一个新的顶级窗口
        result_window.title("识别结果详情")  # 设置窗口标题
        result_window.geometry("600x400")  # 设置窗口大小
        result_window.resizable(False, False)  # 禁止调整窗口大小

        # 主框架
        main_frame = tk.Frame(result_window, padx=20, pady=20)  # 创建主框架
        main_frame.pack(fill=tk.BOTH, expand=True)  # 放置框架

        # 标题
        title_label = tk.Label(main_frame, text="车牌识别结果详情",
                              font=("微软雅黑", 16, "bold"))  # 创建标题
        title_label.pack(pady=(0, 20))  # 放置标题

        # 结果对比框架
        results_frame = tk.Frame(main_frame, relief=tk.RIDGE, bd=2)  # 创建带边框的框架
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)  # 放置框架

        # Tesseract结果
        tesseract_frame = tk.Frame(results_frame, bg="#ecf0f1", padx=15, pady=10)  # 创建Tesseract结果的框架
        tesseract_frame.pack(fill=tk.X, pady=5, padx=5)  # 放置框架

        tk.Label(tesseract_frame, text="Tesseract OCR (本地识别):",
                font=("微软雅黑", 12, "bold"), bg="#ecf0f1").pack(anchor='w')  # 创建标签
        tk.Label(tesseract_frame, text=f"结果: {self.tesseract_result or '未识别出结果'}",
                font=("Arial", 14), bg="#ecf0f1", fg="#2c3e50").pack(anchor='w', pady=5)  # 创建显示结果的标签

        # Gemini结果
        gemini_frame = tk.Frame(results_frame, bg="#fdf2e9", padx=15, pady=10)  # 创建Gemini结果的框架
        gemini_frame.pack(fill=tk.X, pady=5, padx=5)  # 放置框架

        tk.Label(gemini_frame, text="Gemini Pro Vision (云端AI):",
                font=("微软雅黑", 12, "bold"), bg="#fdf2e9").pack(anchor='w')  # 创建标签
        tk.Label(gemini_frame, text=f"结果: {self.gemini_result or '未识别出结果'}",
                font=("Arial", 14), bg="#fdf2e9", fg="#d35400").pack(anchor='w', pady=5)  # 创建显示结果的标签

        # 最终结果
        final_frame = tk.Frame(results_frame, bg="#e8f5e8", padx=15, pady=10)  # 创建最终结果的框架
        final_frame.pack(fill=tk.X, pady=5, padx=5)  # 放置框架

        tk.Label(final_frame, text="最终采用结果:",
                font=("微软雅黑", 12, "bold"), bg="#e8f5e8").pack(anchor='w')  # 创建标签
        final_result = self.result_var.get()  # 获取最终结果
        tk.Label(final_frame, text=f"车牌号: {final_result}",
                font=("Arial", 18, "bold"), bg="#e8f5e8", fg="#27ae60").pack(anchor='w', pady=5)  # 创建显示结果的标签

        # 按钮框架
        button_frame = tk.Frame(main_frame)  # 创建按钮框架
        button_frame.pack(fill=tk.X, pady=(20, 0))  # 放置框架

        # 保存结果按钮
        def save_result():  # 定义保存结果的内部函数
            if final_result and final_result != "---" and final_result != "识别失败":  # 检查是否有有效结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取时间戳
                filename = f"recognition_result_{timestamp}.txt"  # 生成文件名
                try:
                    with open(filename, 'w', encoding='utf-8') as f:  # 打开文件
                        f.write(f"车牌识别结果报告\n")  # 写入标题
                        f.write(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 写入时间
                        f.write(f"Tesseract结果: {self.tesseract_result or '未识别'}\n")  # 写入Tesseract结果
                        f.write(f"Gemini结果: {self.gemini_result or '未识别'}\n")  # 写入Gemini结果
                        f.write(f"最终结果: {final_result}\n")  # 写入最终结果
                        f.write(f"使用引擎: {self.engine_var.get()}\n")  # 写入使用的引擎
                    messagebox.showinfo("保存成功", f"结果已保存到: {filename}")  # 显示成功信息
                except Exception as e:  # 捕获异常
                    messagebox.showerror("保存失败", f"无法保存结果: {str(e)}")  # 显示失败信息
            else:
                messagebox.showwarning("提示", "没有有效的识别结果可以保存")  # 提示用户

        tk.Button(button_frame, text="保存结果", command=save_result,
                 font=("微软雅黑", 12), bg="#3498db", fg="white",
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)  # 创建并放置保存按钮

        tk.Button(button_frame, text="关闭", command=result_window.destroy,
                 font=("微软雅黑", 12), bg="#e74c3c", fg="white",
                 padx=20, pady=5).pack(side=tk.RIGHT, padx=5)  # 创建并放置关闭按钮

    def update_gemini_status(self, status):  # 定义更新Gemini状态显示的方法
        """更新Gemini状态显示"""
        self.gemini_var.set(status)  # 设置Gemini结果标签的文本
        self.root.update_idletasks()  # 强制UI立即更新

    def run_hybrid_recognition(self):  # 定义运行混合识别流程的主方法
        """运行混合识别流程"""
        self.progress['value'] = 0  # 重置进度条

        # 重置结果显示
        self.tesseract_var.set("处理中...")  # 更新Tesseract结果显示
        self.gemini_var.set("处理中...")  # 更新Gemini结果显示
        self.result_var.set("---")  # 重置最终结果显示
        self.engine_var.set("引擎: N/A")  # 重置引擎显示

        try:
            # --- 阶段 1: 图像预处理 ---
            self.status_var.set("阶段1: 图像预处理...")  # 更新状态
            self.log_debug("开始图像预处理")  # 记录日志
            self.progress['value'] = 10  # 更新进度条

            if self.cancel_recognition:  # 检查是否需要取消
                self.cleanup_after_recognition()  # 清理资源
                return  # 退出

            # 灰度处理
            if self.gray_image is None:  # 如果没有缓存
                self.gray_image = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)  # 生成灰度图
                self.log_debug("完成灰度转换")  # 记录日志
            self.progress['value'] = 20  # 更新进度条

            if self.cancel_recognition:  # 检查是否需要取消
                self.cleanup_after_recognition()  # 清理资源
                return  # 退出

            # 二值化处理
            if self.binary_image is None:  # 如果没有缓存
                _, self.binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 生成二值图
                self.log_debug("完成二值化处理")  # 记录日志
            self.progress['value'] = 30  # 更新进度条

            # --- 阶段 2: 车牌定位 ---
            self.status_var.set("阶段2: 车牌定位...")  # 更新状态
            self.log_debug("开始车牌定位")  # 记录日志
            self.plate_image = self.locate_plate(self.original_image_cv)  # 调用车牌定位方法
            if self.plate_image is not None:  # 如果定位成功
                self.log_debug(f"车牌定位成功，尺寸: {self.plate_image.shape}")  # 记录日志
            else:
                self.log_debug("未能定位到车牌区域")  # 记录日志

            self.progress['value'] = 40  # 更新进度条

            if self.cancel_recognition:  # 检查是否需要取消
                self.cleanup_after_recognition()  # 清理资源
                return  # 退出

            # --- 阶段 3: 开始并行识别 ---
            self.status_var.set("阶段3: 同时使用两种引擎识别...")  # 更新状态
            self.log_debug("开始使用两种引擎识别")  # 记录日志

            # 定义要识别的图像
            target_image = self.plate_image if self.plate_image is not None else self.original_image_cv  # 如果定位成功则用定位图，否则用原图
            self.log_debug(f"识别目标图像类型: {'车牌区域' if self.plate_image is not None else '原始图像'}")  # 记录日志

            # 显示目标识别区域
            if self.plate_image is not None:  # 如果定位成功
                self.display_cv_image(self.plate_image, self.processed_label, max_size=(800, 400))  # 显示车牌区域
                self.status_var.set("车牌区域提取成功，开始识别...")  # 更新状态
            else:
                self.display_cv_image(self.original_image_cv, self.processed_label, max_size=(800, 400))  # 显示原始图像
                self.status_var.set("未检测到车牌，尝试整图识别...")  # 更新状态

            # --- 阶段 3a: Tesseract识别 ---
            self.progress['value'] = 50  # 更新进度条
            if self.cancel_recognition:  # 检查是否需要取消
                self.cleanup_after_recognition()  # 清理资源
                return  # 退出

            self.tesseract_result = self.recognize_with_tesseract(target_image)  # 调用Tesseract识别
            self.tesseract_var.set(self.tesseract_result if self.tesseract_result else "未识别出结果")  # 更新UI
            self.progress['value'] = 60  # 更新进度条

            # --- 阶段 3b: Gemini API识别 ---
            if self.cancel_recognition:  # 检查是否需要取消
                self.cleanup_after_recognition()  # 清理资源
                return  # 退出

            self.progress['value'] = 70  # 更新进度条
            gemini_thread = threading.Thread(target=self.gemini_recognition_thread, args=(target_image,))  # 创建Gemini识别线程
            gemini_thread.start()  # 启动线程

            # 等待Gemini API线程，最多30秒
            start_time = time.time()  # 记录开始时间
            while gemini_thread.is_alive():  # 当线程仍在运行时
                elapsed = time.time() - start_time  # 计算已用时间
                if elapsed > 30:  # 如果超过30秒
                    self.log_debug("Gemini API请求超时")  # 记录日志
                    self.gemini_var.set("请求超时")  # 更新UI
                    break  # 退出等待

                if self.cancel_recognition:  # 检查是否需要取消
                    self.log_debug("用户取消了识别过程")  # 记录日志
                    self.cleanup_after_recognition()  # 清理资源
                    return  # 退出

                # 更新进度条显示请求进度
                self.update_gemini_status(f"请求中... ({int(elapsed)}s)")  # 更新UI状态
                time.sleep(0.5)  # 等待0.5秒

            gemini_thread.join(1)  # 再给1秒钟完成

            self.progress['value'] = 80  # 更新进度条

            # --- 阶段 4: 结果评估和选择 ---
            self.status_var.set("阶段4: 结果评估和最佳选择...")  # 更新状态
            self.log_debug("开始评估和选择最佳结果")  # 记录日志
            self.progress['value'] = 90  # 更新进度条

            # 优先使用Tesseract结果，如果符合车牌格式要求
            if self.tesseract_result and len(self.tesseract_result) >= 6:  # 检查Tesseract结果是否有效
                self.result_var.set(self.tesseract_result)  # 采纳Tesseract结果
                self.engine_var.set("引擎: Tesseract OCR (本地)")  # 更新引擎信息
                self.status_var.set("本地识别成功!")  # 更新状态
                self.log_debug(f"最终选择Tesseract结果: {self.tesseract_result}")  # 记录日志
            # 次优先使用Gemini结果，如果可用
            elif self.gemini_result:  # 检查Gemini结果是否有效
                self.result_var.set(self.gemini_result)  # 采纳Gemini结果
                self.engine_var.set("引擎: Gemini Pro Vision (云端)")  # 更新引擎信息
                self.status_var.set("Gemini API 识别成功!")  # 更新状态
                self.log_debug(f"最终选择Gemini结果: {self.gemini_result}")  # 记录日志
            # 都失败的情况
            else:
                self.result_var.set("识别失败")  # 设置为失败
                self.engine_var.set("引擎: N/A")  # 更新引擎信息
                self.status_var.set("所有方法均识别失败。")  # 更新状态
                self.log_debug("所有方法均识别失败")  # 记录日志

            self.progress['value'] = 100  # 更新进度条至100%

            # 显示详细结果对话框
            self.show_detailed_results()  # 调用方法显示详细结果

        except Exception as e:  # 捕获整个过程中的异常
            self.log_debug(f"识别过程中出现异常: {str(e)}")  # 记录日志
            self.log_debug(traceback.format_exc())  # 记录详细堆栈跟踪
            messagebox.showerror("错误", f"识别过程中出现异常: {str(e)}")  # 显示错误提示框
        finally:
            self.cleanup_after_recognition()  # 无论成功与否，最后都执行清理操作

    def gemini_recognition_thread(self, image):  # 定义在单独线程中运行Gemini识别的方法
        """在单独的线程中运行Gemini API识别"""
        try:
            self.log_debug("Gemini识别线程开始")  # 记录日志
            gemini_response = self.recognize_with_gemini(image)  # 调用Gemini识别方法
            self.log_debug(f"Gemini返回结果: {gemini_response}")  # 记录返回结果

            if gemini_response.get('status') == 'success':  # 检查API调用是否成功
                self.gemini_result = gemini_response.get('plate_number', 'N/A')  # 获取车牌号码
                self.gemini_var.set(self.gemini_result)  # 更新UI
                self.log_debug(f"Gemini识别成功: {self.gemini_result}")  # 记录日志
            else:
                reason = gemini_response.get('reason', '未知错误')  # 获取失败原因
                self.gemini_var.set(f"失败: {reason}")  # 更新UI
                self.gemini_result = None  # 将结果设为None
                self.log_debug(f"Gemini识别失败: {reason}")  # 记录日志
        except Exception as e:  # 捕获线程中的异常
            self.log_debug(f"Gemini线程异常: {str(e)}")  # 记录日志
            self.log_debug(traceback.format_exc())  # 记录详细堆栈跟踪
            self.gemini_var.set(f"错误: {str(e)}")  # 更新UI
            self.gemini_result = None  # 将结果设为None

    def cleanup_after_recognition(self):  # 定义识别结束后的清理方法
        """清理识别过程资源"""
        self.cancel_button.config(state=tk.DISABLED)  # 禁用取消按钮
        if self.cancel_recognition:  # 如果是因取消而结束
            self.progress['value'] = 0  # 重置进度条
            self.status_var.set("识别已取消")  # 更新状态
            self.log_debug("识别过程已取消并清理")  # 记录日志

if __name__ == "__main__":  # Python程序的入口点
    root = tk.Tk()  # 创建Tkinter的主窗口
    app = HybridLPR_App(root)  # 实例化主应用程序类
    root.mainloop()  # 进入Tkinter事件循环，等待用户操作

# ====================================================================================================
#
#                                           项目代码总体说明
#
# 本项目是一个基于Python的混合视觉模型车牌识别系统，使用了Tkinter构建图形用户界面（GUI）。
# 其核心功能是结合了两种不同的车牌识别技术：
# 1. 本地OCR引擎：使用Tesseract OCR，通过一系列图像预处理（灰度化、降噪、二值化、形态学操作）来提升识别准确率。
#    这种方法运行在本地，速度快，不依赖网络。
# 2. 云端AI模型：使用Google的Gemini Pro Vision多模态大模型，将图像发送到云端进行分析和识别。
#    这种方法利用了先进的AI能力，通常在图像质量较差或复杂背景下表现更好，但需要网络连接和API密钥。
#
# 主要功能模块：
# - 用户界面 (UI)：
#   - 使用Tkinter构建，包含一个登录界面和主操作界面。
#   - 主界面分为左侧控制面板和右侧图像显示区。
#   - 控制面板提供文件选择（图片/视频）、图像处理预览、识别控制（开始/取消）、结果展示等功能。
#   - 图像显示区可以展示原始图像和处理后的图像（如定位到的车牌区域）。
#
# - 图像/视频处理：
#   - 支持加载多种格式的图片和视频文件。
#   - 对视频文件可以进行播放、暂停和停止操作。
#   - 包含一个初步的车牌定位算法（locate_plate），该算法结合了边缘检测、轮廓分析和颜色分析（查找蓝色区域）来尝试从原始图像中裁剪出车牌部分，以提高后续识别的准确性。
#
# - 混合识别流程 (run_hybrid_recognition)：
#   - 这是系统的核心逻辑，在一个单独的线程中运行以避免UI卡顿。
#   - 流程包括：图像预处理 -> 车牌定位 -> 并行调用Tesseract和Gemini进行识别 -> 结果评估与选择。
#   - Tesseract识别会尝试多种预处理组合，并根据一套评分系统（evaluate_plate_text）选出最佳结果。
#   - Gemini识别通过API调用，并设置了超时机制。
#   - 最终系统会优先采纳格式更规范的Tesseract结果，如果Tesseract失败则采纳Gemini的结果。
#
# - 结果展示与调试：
#   - 界面上会分别显示Tesseract和Gemini的识别结果，以及最终采纳的结果和所使用的引擎。
#   - 提供一个详细的结果弹窗，清晰对比两种方法的结果，并可以保存识别报告。
#   - 内置了详细的日志系统（logging和自定义的log_debug），可以记录每一步操作的详细信息，并提供一个“显示调试信息”的窗口，方便开发者排查问题。
#
# - 健壮性与用户体验：
#   - 大量使用try-except块来捕获和处理可能发生的错误（如文件读取失败、API调用异常等），并通过messagebox向用户反馈。
#   - 耗时操作（如识别过程）均在后台线程中执行，并通过进度条和状态栏向用户展示进度。
#   - 提供了取消功能，允许用户中断正在进行的识别任务。
#
# 如何运行：
# 1. 确保已安装所有必要的Python库（如opencv-python, pillow, pytesseract, google-generativeai等）。
# 2. 根据实际情况，在代码中配置Tesseract OCR的路径 (`pytesseract.pytesseract.tesseract_cmd`)。
# 3. 在代码中填入有效的Google Gemini API密钥 (`API_KEY`)。
# 4. 运行 `main.py` 文件。
#
# ====================================================================================================