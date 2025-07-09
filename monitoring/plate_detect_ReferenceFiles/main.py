# coding=gbk

import cv2
from tkinter.filedialog import askopenfilename
import ttkbootstrap as tk
from PIL import Image, ImageTk
import img_function as predict
import img_math as img_math
import img_recognition as img_rec
from chuli import App
from tkinter.messagebox import showerror
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
from PIL import ImageFont
from PIL import ImageDraw
# ����������
import hyperlpr3 as lpr3
class UI_main(ttk.Frame):
    pic_path = ""   #ͼƬ·��
    pic_source = "" 
    colorimg = 'white'   #������ɫ
    cameraflag = 0
    width = 700   #��
    height = 400   #��
    flag=0
    base_server_data=['��Ե���','AIʶ��']
    color_transform = img_rec.color_tr

    def __init__(self, win):
        ttk.Frame.__init__(self, win)

        
        win.title("����ʶ��ϵͳ")
        win.geometry('+300+200')
        win.minsize(UI_main.width,UI_main.height)
        win.configure(relief=tk.RIDGE)
        win.themename="litera"
        # win.update()
        self.font_ch = ImageFont.truetype("./pic/simfang.ttf", 19, 0)
 
        # ʵ����ʶ�����
        self.catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)

        self.pack(fill=tk.BOTH)
        frame_left = ttk.Frame(self)
        frame_right_1 = ttk.Frame(self)
        frame_right_2 = ttk.Frame(self)
        frame_left.pack(side=tk.LEFT, expand=1, fill=tk.BOTH)
        frame_right_1.pack( expand=1)
        frame_right_2.pack()

        #������� --->����ʶ���������ͼƬ
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack()

        #�����ұ� --->��λ����λ�á�ʶ����
        ttk.Label(frame_right_1, text='��λ���ƣ�',bootstyle=SUCCESS, font=('Times', '14')).grid(
            column=0, row=6, sticky=tk.NW)

        
        self.roi_ct2 = ttk.Label(frame_right_1)
        self.roi_ct2.grid(column=0, row=7, columnspan=2,sticky=tk.W,pady=5)
        ttk.Label(frame_right_1, text='ʶ������',bootstyle=PRIMARY,font=('Times', '14')).grid(
            column=0, row=8, sticky=tk.W,pady=5)
        self.r_ct2 = ttk.Label(frame_right_1, text="", font=('Times', '20'))
        self.r_ct2.grid(column=0, row=9, sticky=tk.W,pady=5,columnspan=2)
        
        #������ɫ

        
        #������ѡ��ʶ�𷽷�

        self.color_ct2 = ttk.Label(frame_right_1,background=self.colorimg, 
                        bootstyle=PRIMARY,text="ʶ�𷽷�:", width="15",font=('Times', '14'))
        self.color_ct2.grid(column=0, row=10,sticky=tk.W,columnspan=2)

        
        
        self.select_text_data = tk.StringVar()
        self.select_box_obj = ttk.Combobox(
            master=frame_right_1,
            text='',
            textvariable=self.select_text_data,
            font=('΢���ź�', 10),
            values=self.base_server_data,  # �������ֵ
            height=10,  # �߶�
            width=9,  # ���
            state='readonly',  # ����״̬ normal(��ѡ������)��readonly(ֻ��ѡ)�� disabled
            cursor='arrow',  # ����ƶ�ʱ��ʽ arrow, circle, cross, plus...
            postcommand=self.set_value_before_choose  # ѡ��ǰ�����ص�
        )
        self.select_box_obj.grid(column=1, row=10,sticky=tk.W,columnspan=2)
        
        self.select_box_obj.bind('<<ComboboxSelected>>',self.submit_result)


        #�������½�
        from_pic_ctl = ttk.Button(
            frame_right_2, text="����ͼƬ", width=20, command=self.from_pic, bootstyle=(SUCCESS, "outline-toolbutton"))
        # from_pic_ctl =       ttk.Button(self, text="Button 1", bootstyle=SUCCESS).pack(side=LEFT, padx=5, pady=10)
        from_pic_ctl.grid(column=0, row=1,columnspan=2,pady=10)



        
        #���ʶ������
        from_pic_chu = ttk.Button(
            frame_right_2, text="���ʶ������", width=20, command=self.clean,bootstyle=(PRIMARY, "outline-toolbutton"))
        from_pic_chu.grid(column=0, row=2,columnspan=2,pady=10)
        #�鿴ͼ�������
        from_pic_chu = ttk.Button(
            frame_right_2, text="�鿴ͼ�������", width=20, command=self.pic_chuli,bootstyle=(INFO, "outline-toolbutton"))
        from_pic_chu.grid(column=0, row=3,columnspan=2,pady=10)

        

        self.clean()

        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

    def set_value_before_choose(self):
        """
        ѡ��ǰ�����ı��������ɸѡ��������������
        :return:
        """
        # print('ѡ��ǰ�������ֵ��', self.select_text_data.get())
        return

    def submit_result(self,event):
        print('��ǰѡ��{}'.format(self.select_box_obj.get()))
        if(self.select_box_obj.get()=='��Ե���'):
            self.flag=0
        else:
            self.flag=1
    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        pil_image_resized = im.resize((500,400),Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=pil_image_resized)
        return imgtk

    #��ʾͼƬ�������
    def pic_chuli(self):
        # os.system("python3.8 ./chuli.py")
        app = App()

    def pic(self, pic_path):
        img_bgr = img_math.img_read(pic_path)
        first_img, oldimg = self.predictor.img_first_pre(img_bgr)
        if not self.cameraflag:
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
        r_color, roi_color, color_color = self.predictor.img_only_color(oldimg, 
                                                                oldimg, first_img)
        # self.color_ct2.configure(background=color_color)
        try:
            Plate = HyperLPR_PlateRecogntion(img_bgr)
            r_color = Plate[0][0]
        except:
            pass
        if(self.flag==0):

            self.show_roi(r_color, roi_color)
            self.colorimg = color_color
            print("|", color_color,
                r_color, "|", self.pic_source)
        else:
            results = self.catcher(img_bgr)                  #AI��ⷽ��
            for code, confidence, type_idx, box in results:
                # �������ݲ�����
                text = f"{code} - {confidence:.2f}"
                image = self.draw_plate_on_image(img_bgr, box, text, font=self.font_ch)
                self.imgtk = self.get_imgtk(image)
                self.image_ctl.configure(image=self.imgtk)
                left_up_x,left_up_y,right_down_x,right_down_y=box
                crop = img_bgr[int(left_up_y):int(right_down_y), int(left_up_x):int(right_down_x)]  #�ü���ĳ���ͼƬ
                self.show_roi(code, crop)                                                      #���ƺ���ͼƬ��ʾ



    def draw_plate_on_image(self,img, box, text, font):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (139, 139, 102), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (139, 139, 102), -1)
        data = Image.fromarray(img)
        draw = ImageDraw.Draw(data)
        draw.text((x1 + 1, y1 - 18), text, (0, 255, 0), font=font)
        res = np.asarray(data)
 
        return res
    #����ͼƬ--->��ϵͳ�ӿڻ�ȡͼƬ����·��
    def from_pic(self):
        self.cameraflag = 0
        self.pic_path = askopenfilename(title="ѡ��ʶ��ͼƬ", filetypes=[(
            "ͼƬ", "*.jpg;*.jpeg;*.png")])
        
        self.clean()
        self.pic_source = "�����ļ���" + self.pic_path
        self.pic(self.pic_path)
        print(self.colorimg)
           
    def show_roi(self, r, roi):  # ���ƶ�λ���ͼƬ
        if r:
            try:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = Image.fromarray(roi)
                pil_image_resized = roi.resize((200, 50), Image.ANTIALIAS)
                self.tkImage2 = ImageTk.PhotoImage(image=pil_image_resized)
                self.roi_ct2.configure(image=self.tkImage2, state='enable')
            except:
                pass
            self.r_ct2.configure(text=str(r))
            # try:
            #     c = self.color_transform[color]
            #     self.color_ct2.configure(text=c[0], state='enable')
            # except:
            #     self.color_ct2.configure(state='disabled')
   
    #���ʶ������,��ԭ��ʼ���
    def clean(self):
        img_bgr3 = img_math.img_read("source/pic/hy.png")
        self.imgtk2 = self.get_imgtk(img_bgr3)
        self.image_ctl.configure(image=self.imgtk2)

        self.r_ct2.configure(text="")
        # self.color_ct2.configure(text="", state='enable')
        #��ʾ������ɫ
        # self.color_ct2.configure(background='white' ,text="ѡ��", state='enable')
        self.pilImage3 = Image.open("source/pic/locate.png")
        pil_image_resized = self.pilImage3.resize((200, 50), Image.ANTIALIAS)
        self.tkImage3 = ImageTk.PhotoImage(image=pil_image_resized)
        self.roi_ct2.configure(image=self.tkImage3, state='enable')
        # shutil.rmtree('./tmp')
        # os.mkdir('./tmp')


if __name__ == '__main__':


    style = ttk.Style(theme='litera')
    win = style.master
    # win = tk.Tk()
    ui_main= UI_main(win)
    # ������Ϣѭ��
    win.mainloop()

