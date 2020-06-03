# coding: utf-8

"""
modified from https://blog.csdn.net/u013232740/article/details/47132735
"""

import os
import tkinter as tk
import cv2

from PIL import Image, ImageTk, ImageFont, ImageDraw # pillow module

def load_image(im_pth):
    im = cv2.imread(im_pth)
    im = im[:, :, ::-1]  # bgr => rgb
    return ImageTk.PhotoImage(Image.fromarray(im))

class COCO_Viewer(tk.Frame):
    """ 定义GUI的应用程序类，派生于Frmae """
    def __init__(self,master=None):
        """ 构造函数，master为父窗口 """
        self.im_dir = 'E:/data/VOC2007/JPEGImages'
        self.files = os.listdir(self.im_dir)   # 获取图像文件名列表
        self.index = 0      # 图片索引，初始显示第一张
        self.img = self.load_image_by_index()
        tk.Frame.__init__(self,master)    #调用父类的构造函数
        self.pack()         # 调整显示的位置和大小
        self.createWidget() # 类成员函数，创建子组件

    def createWidget(self):
        self.lblImage=tk.Label(self,width=300, height=300)  # 创建Label组件以显示图像
        self.lblImage['image']=self.img     # 显示第一张照片
        self.lblImage.pack()                # 调整显示位置和大小
        self.f=tk.Frame()                   # 创建窗口框架
        self.f.pack()
        self.btnPrev=tk.Button(self.f,text="Prev",command=self.prev)  # 创建按钮
        self.btnPrev.pack(side=tk.LEFT)
        self.btnNext=tk.Button(self.f,text="Next",command=self.next)  # 创建按钮
        self.btnNext.pack(side=tk.LEFT)

    def load_image_by_index(self):
        im_pth = self.im_dir + '/' + self.files[self.index]
        im = load_image(im_pth)
        return im

    def prev(self):  # 事件处理函数
        self.showfile(-1)

    def next(self):   # 事件处理函数
        self.showfile(1)

    def showfile(self,n):
        self.index += n
        if self.index<0:
            self.index=len(self.files)-1    # 循环显示最后一张
        if self.index>len(self.files)-1:
            self.index=0    # 循环显示第一张
        self.img = self.load_image_by_index()
        self.lblImage['image']=self.img

def main():
    root = tk.Tk()   # 创建一个Tk根窗口组件
    root.title("Simple Image Browser")   # 设置窗口标题
    app = COCO_Viewer(master=root)   # 创建Application的对象实例
    app.mainloop()   # 事件循环

if __name__ == '__main__':
    main()

