#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

__author__ = 'Zhuo Zhang'
__copyright__ = 'Copyright 2017-2020, Zhuo Zhang'
__license__ = 'MIT'
__version__ = '0.1'
__email__ = 'imzhuo@foxmail.com'
__status__ = 'Development'
__description__ = 'Tkinter based GUI, visualizing PASCAL VOC object detection annotation'

"""
Changelog:
- 2020-06-01 14:44:01
    Draw object class name. Add license. Polish meta info.

- 2017.10.22 22:36
    Created project. Dependencies: Python, Tkinter(GUI), opencv(image processing), 
    lxml(annotation parsing).
    You may need this: pip install --upgrade image pillow lxml numpy
"""

try:
    import Tkinter as tk
except:
    import tkinter as tk
from PIL import Image, ImageTk, ImageFont, ImageDraw # pillow module
import os
try:
    from tkFileDialog import askdirectory
except:
    from tkinter.filedialog import askdirectory
import cv2
from lxml import etree
import numpy as np


def zz_draw_text(im, text, text_org, font):
    """
    Draw text on OpenCV's Image (ndarray)
    Implemented by: ndarray -> pil's image -> draw text -> ndarray

    Note: OpenCV puttext's drawback: font too large, no anti-alias, can't show Chinese chars

    @param im: opencv loaded image
    @param text: text(string) to be put. support Chinese
    @param font: font, e.g. ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', font_size)

    Example Usage:
    font_size = 20
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', font_size)
    text_org = (256, 256)
    im = zz_draw_text(im, "object", text_org, font)
    """
    im_pil = Image.fromarray(im)
    draw = ImageDraw.Draw(im_pil)
    draw.text(text_org, text, font=font, fill=(0, 0, 255, 0))
    output = np.array(im_pil)
    return output


class BndBox(object):
    def __init__(self, x1=0, y1=0, x2=0, y2=0, cls_name=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls_name = cls_name # class name


class PascalVOC2007XML:
    def __init__(self, xml_full_name):
        # TODO:校验xml_full_name这个文件的合法性
        self.tree = etree.parse(xml_full_name)
        self.boxes = []

    def get_boxes(self):
        if len(self.boxes) == 0:
            for obj in self.tree.xpath('//object'):
                box = BndBox()
                for item in obj.getchildren():
                    if (item.tag=='name'):
                        box.cls_name = item.text
                    elif (item.tag=='bndbox'):
                        coords = [int(float(_.text)) for _ in item.getchildren()]
                        box.x1, box.y1, box.x2, box.y2 = coords
                self.boxes.append(box)
        return self.boxes


class VOC_Viewer(tk.Tk):
    def __init__(self, im_dir=None, show_x=None, show_y=None, box_thick=1):
        # 加载图像：tk不支持直接使用jpg图片。需要Pillow模块进行中转
        """
        @param im_dir: 包含图片的路径，也就是"JPEGImages". 要求它的同级目录中包含Annotations目录，里面包含各种xml文件。
        @param show_x: 图片显示时候的宽度
        @param show_y: 图片显示时的高度
        @param box_thick: 画框的宽度
        """
        super().__init__()

        # set title and window size
        self.title('imageset viewer')
        self.width = (int)(0.7 * self.winfo_screenwidth())
        self.height = (int)(0.7 * self.winfo_screenheight())
        self.geometry('%dx%d' % (self.width, self.height))

        # custom settings
        self.show_x = show_x
        self.show_y = show_y
        self.box_thick = box_thick

        self.im_dir = tk.StringVar()
        self.path_entry = tk.Entry(self, text=self.im_dir, width=60, state='readonly')
        self.path_entry.grid(row=0, column=0)

        self.choose_path_btn = tk.Button(self, text='图片路径', command=self.selectPath)
        self.choose_path_btn.grid(row=0, column=1)

        # Surface image
        self.tkim = self.get_surface_tkim()
        # You may also change the surface with your favorite picture
        # im_name = '/home/chris/Pictures/im/girl2.jpg'
        # self.tkim = ImageTk.PhotoImage(Image.open(im_name))

        self.labelPic = tk.Label(self, justify='left',
                          image=self.tkim, compound='center',
                          fg='white', bg='white',
                          width = self.width-250, height = self.height-50)
        self.labelPic.grid(row=1, column=0)

        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)

        self.listbox = tk.Listbox(self, yscrollcommand=self.scrollbar.set)
        self.listbox.grid(row=1, column=2, sticky=tk.N+tk.S)

        self.im_names = []
        if im_dir is not None:
            self.im_dir.set(im_dir)
            # Get natural file list, same as Windows explorer's result
            self.im_names = [_ for _ in os.listdir(self.im_dir.get())]
            self.im_names.sort()
            for im_name in self.im_names:
                self.listbox.insert(tk.END, im_name)
        self.listbox.bind('<<ListboxSelect>>', self.callback)
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.grid(row=1, column=3, sticky=tk.N+tk.S)

    def callback(self, event=None):
        im_id = self.listbox.curselection()
        if im_id:
            im_name = self.listbox.get(im_id)
            if (im_name.endswith('.jpg') or im_name.endswith('.png')):
                im_name_full = os.path.join(self.im_dir.get(), im_name).replace('\\', '/')
                self.tkim = self.get_tkim(im_name_full)
                self.labelPic.configure(image=self.tkim)
                # print(im_name_full)

    def get_tkim(self, im_name_full):
        """
        读取图像并转化为tkim。必要时做resize
        """
        im = cv2.imread(im_name_full)
        print('reading image:', im_name_full)
        im_ht, im_wt, im_dt = im.shape
        show_x = self.show_x
        show_y = self.show_y
        if show_x is None:
            show_x = im_wt
        if show_y is None:
            show_y = im_ht
        if show_x!=im_wt or show_y!=im_ht:
            im = cv2.resize(im, (show_x, show_y))
            print('doing resize!')
            print('show_x={:d}, im_wt={:d}, show_y={:d}, im_ht={:d}'.format(show_x, im_wt, show_y, im_ht))
        scale_x = im_wt*1.0 / show_x
        scale_y = im_ht*1.0 / show_y
        anno_name_full = im_name_full.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml').replace('.png', '.xml')
        print('anno_name_full is:', anno_name_full)
        if os.path.exists(anno_name_full):
            print(' existing the xml file!')
            boxes = self.get_boxes_from_voc_xml(anno_name_full)
            for box in boxes:
                xmin = int(box.x1/scale_x)
                ymin = int(box.y1/scale_y)
                xmax = int(box.x2/scale_x)
                ymax = int(box.y2/scale_y)
                cv2.rectangle(im,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=(0, 255, 0),
                          thickness=self.box_thick
                          )
                font_size = 16
                font = ImageFont.truetype('‪C:/Windows/Fonts/msyh.ttc', font_size)
                text_org = (xmin+10, ymin+10)
                im = zz_draw_text(im, box.cls_name, text_org, font)

        im = im[:, :, ::-1]  # bgr => rgb   necessary
        tkim = ImageTk.PhotoImage(Image.fromarray(im))
        return tkim

    def get_surface_tkim(self):
        """封面图片"""
        im = np.ndarray((500, 700, 3), dtype=np.uint8)

        cv2.putText(im, 'Please choose image set folder',
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color=(255,255,255)
                    )
        im = im[:, :, ::-1]

        tkim = ImageTk.PhotoImage(image=Image.fromarray(im))
        return tkim

    def get_boxes_from_voc_xml(self, anno_name_full):
        anno = PascalVOC2007XML(anno_name_full)
        boxes = anno.get_boxes()
        return boxes

    def selectPath(self):
        pth = askdirectory()

        # 清空listbox中的元素
        self.listbox.delete(0, len(self.im_names)-1)

        self.fill_im_names(pth)

    def fill_im_names(self, im_dir):
        if im_dir is not None:
            self.im_dir.set(im_dir)
            # 获取自然顺序的文件列表
            self.im_names = [_ for _ in os.listdir(im_dir)]
            self.im_names.sort()
            for im_name in self.im_names:
                self.listbox.insert(tk.END, im_name)

if __name__ == '__main__':
    ## 最简单的方式：不预设im_dir，打开GUI后自行选择图片路径
    app = VOC_Viewer(im_dir=None, box_thick=2)

    """
    ## 也可以在代码中指定
    ## eg1: 指定图片路径
    im_dir = '/opt/data/PASCAL_VOC/VOCdevkit2007/TT100/JPEGImages'
    app = App(root, im_dir)

    ## eg2: 还可以指定显示的图片的长度和宽度，也就是要做图像缩放了。
    app = App(root, im_dir, show_x=1000, show_y=1000)

    ## eg3: 指定画框的宽度
    app = App(root, im_dir, box_thick=2)
    # 或者更多的指定：
    app = App(root, im_dir, show_x=1000, show_y=1000, box_thick=2)
    """

    # message loop
    app.mainloop()
