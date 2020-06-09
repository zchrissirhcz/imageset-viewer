#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

__author__ = 'Zhuo Zhang'
__copyright__ = 'Copyright 2017-2020, Zhuo Zhang'
__license__ = 'MIT'
__version__ = '0.3'
__email__ = 'imzhuo@foxmail.com'
__status__ = 'Development'
__description__ = 'Tkinter based GUI, visualizing PASCAL VOC object detection annotation'

"""
Changelog:

- 2020-06-09 23:14   v0.3
    User select saving directory(optional) for picking up interested images.
    By pressing left control button, selected image is saved.

- 2020-06-02 16:40   v0.2
    User choose image and annotation folders separately. Better UI layout.
    Colorful boxes and class name text.

- 2020-06-01 14:44  v0.1
    Draw object class name. Add license. Polish meta info. Adjust UI.

- 2017.10.22 22:36  v0.0
    Created project. Dependencies: Python, Tkinter(GUI), opencv(image processing), 
    lxml(annotation parsing).
    You may need this: pip install --upgrade image pillow lxml numpy
"""

from PIL import Image, ImageTk, ImageFont, ImageDraw # pillow module
import os
import cv2
from lxml import etree
import numpy as np
import random
import colorsys
import shutil

try: # py3
    import tkinter as tk
    from tkinter.filedialog import askdirectory
except: # py2
    import Tkinter as tk
    from tkFileDialog import askdirectory


def draw_text(im, text, text_org, color=(0,0,255,0), font=None):
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
    im = draw_text(im, "object", text_org, font)
    """
    im_pil = Image.fromarray(im)
    draw = ImageDraw.Draw(im_pil)
    draw.text(text_org, text, font=font, fill=color)
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
    def __init__(self, xml_pth):
        # TODO: validate xml_pth's content
        self.tree = etree.parse(xml_pth)
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


def get_color_table(num_cls=20):
    # num_cls: number of classes.  20 for voc, 80 for coco
    hsv_tuples = [(x / num_cls, 1., 1.) for x in range(80)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(42)
    random.shuffle(colors)
    random.seed(None)
    return colors


class VOC_Viewer(tk.Tk):
    def __init__(self, im_dir=None, show_x=None, show_y=None, box_thick=1):
        # 加载图像：tk不支持直接使用jpg图片。需要Pillow模块进行中转
        """
        @param im_dir: 包含图片的路径，也就是"JPEGImages". 要求它的同级目录中包含Annotations目录，里面包含各种xml文件。
        @param show_x: 图片显示时候的宽度
        @param show_y: 图片显示时的高度
        @param box_thick: thickness of bounding box
        """
        super().__init__()

        # custom settings
        self.show_x = show_x
        self.show_y = show_y
        self.box_thick = box_thick
        self.bg = '#34373c'
        self.fg = '#f2f2f2'

        # set title, window size and background
        self.title('ImageSet Viewer ' + __version__)
        self.width = (int)(0.6 * self.winfo_screenwidth())
        self.height = (int)(0.6 * self.winfo_screenheight())
        self.geometry('%dx%d+200+100' % (self.width, self.height))
        self.configure(bg=self.bg)
        self.minsize(800, 600)

        self.init_components(im_dir)
        self.init_dataset()
    
    def init_dataset(self):
        self.cls_names = [ #'__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.num_classes = len(self.cls_names)
        self.color_table = get_color_table(self.num_classes)
        self.class_to_ind = dict(zip(self.cls_names, range(self.num_classes)))
    
    def get_color_by_cls_name(self, cls_name):
        ind = self.class_to_ind[cls_name]
        return self.color_table[ind]

    def init_components(self, im_dir):
        # 设置顶级窗体的行列权重，否则子组件的拉伸不会填充整个窗体
        # ref: https://blog.csdn.net/acaic/article/details/80963688
        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=1)

        # Top Level Layout: main_frame & side_frame
        main_frame = tk.LabelFrame(self, bg=self.bg)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)

        side_frame = tk.LabelFrame(self, bg=self.bg)
        side_frame.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NSEW)

        # main_frame: directory_frame & image_frame
        main_frame.rowconfigure(0, weight=20)
        main_frame.rowconfigure(1, weight=80)
        main_frame.columnconfigure(0, weight=1)

        directory_frame = tk.LabelFrame(main_frame, bg=self.bg)
        directory_frame.grid(row=0, column=0, sticky=tk.NSEW)

        image_frame_height = (int)(0.7*self.height)
        image_frame = tk.LabelFrame(main_frame, height=image_frame_height, bg=self.bg)
        image_frame.grid(row=1, column=0, sticky=tk.NSEW)
        # 使组件大小不变 https://zhidao.baidu.com/question/1643979034294549180.html
        image_frame.grid_propagate(0)

        # image_frame
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        self.surface = self.get_surface_image() # Surface image
        # self.surface = self.cv_to_tk(cv2.imread('surface.jpg')) # Use image file
        self.image_label = tk.Label(image_frame, image=self.surface,
                          bg=self.bg, fg=self.fg,compound='center')
        self.image_label.grid(row=0, column=0, sticky=tk.NSEW)

        # side_frame
        side_frame.rowconfigure(0, weight=5)
        side_frame.rowconfigure(1, weight=95)

        image_names_label = tk.Label(side_frame, text="Image Files", bg=self.bg, fg=self.fg)
        image_names_label.grid(row=0, column=0)

        self.scrollbar = tk.Scrollbar(side_frame, orient=tk.VERTICAL)

        self.listbox = tk.Listbox(side_frame, yscrollcommand=self.scrollbar.set)
        self.listbox.grid(row=1, column=0, sticky=tk.NS)

        # directory_frame
        directory_frame.rowconfigure(0, weight=5)
        directory_frame.rowconfigure(1, weight=5)
        directory_frame.rowconfigure(2, weight=5)
        directory_frame.columnconfigure(0, weight=1)
        directory_frame.columnconfigure(1, weight=9)

        # im_dir button
        choose_im_dir_btn = tk.Button(directory_frame, text='Image Directory',
            command=self.select_image_directory, bg=self.bg, fg=self.fg)
        choose_im_dir_btn.grid(row=0, column=0, sticky=tk.NSEW)

        self.im_dir = tk.StringVar()
        im_dir_entry = tk.Entry(directory_frame, text=self.im_dir, state='readonly')
        im_dir_entry.grid(row=0, column=1, sticky=tk.NSEW)

        self.im_names = []
        if im_dir is not None:
            self.im_dir.set(im_dir)
            self.im_names = [_ for _ in os.listdir(self.im_dir.get())]
            self.im_names.sort()
            for im_name in self.im_names:
                self.listbox.insert(tk.END, im_name)
        self.listbox.bind('<<ListboxSelect>>', self.callback)
        # more key binds see https://www.cnblogs.com/muziyunxuan/p/8297536.html
        self.listbox.bind('<Control_L>', self.save_image)
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.grid(row=1, column=1, sticky=tk.NS)

        # anno_dir button
        choose_anno_dir_bn = tk.Button(directory_frame, text='Annotation Directory',
            command=self.select_annotation_directory, bg=self.bg, fg=self.fg)
        choose_anno_dir_bn.grid(row=1, column=0, sticky=tk.NSEW)

        self.anno_dir = tk.StringVar()
        anno_dir_entry = tk.Entry(directory_frame, text=self.anno_dir, state='readonly')
        anno_dir_entry.grid(row=1, column=1, sticky=tk.NSEW)

        # copy (save) dir button
        choose_save_dir_btn = tk.Button(directory_frame, text='Copy Save Directory',
            command=self.select_save_directory, bg=self.bg, fg=self.fg)
        choose_save_dir_btn.grid(row=2, column=0, sticky=tk.NSEW)

        self.save_dir = tk.StringVar()
        save_dir_entry = tk.Entry(directory_frame, text=self.save_dir, state='readonly')
        save_dir_entry.grid(row=2, column=1, sticky=tk.NSEW)

    def callback(self, event=None):
        im_id = self.listbox.curselection()
        if im_id:
            print('---- im_id is: ', im_id)
            im_name = self.listbox.get(im_id)
            if (im_name.endswith('.jpg') or im_name.endswith('.png')):
                im_pth = os.path.join(self.im_dir.get(), im_name).replace('\\', '/')
                self.tkim = self.get_tkim(im_pth)
                self.image_label.configure(image=self.tkim)
                #print(im_pth)

    def save_image(self, event):
        """保存（拷贝）选中的图片到目录
        当前设定为，按左Control键，把当前浏览的图片存储到指定的保存路径。用于手工挑选图片
        """
        im_id = self.listbox.curselection()
        if im_id:
            im_name = self.listbox.get(im_id)
            if (im_name.endswith('.jpg') or im_name.endswith('.png')):
                im_pth = os.path.join(self.im_dir.get(), im_name).replace('\\', '/')
                save_pth = os.path.join(self.save_dir.get(), im_name).replace('\\', '/')
                shutil.copyfile(im_pth, save_pth)
                print('Save(copy) to ' + save_pth)
                #print(im_pth)

    def get_tkim(self, im_pth):
        """
        Load image and annotation, draw on image, and convert to image.
        When necessary, image resizing is utilized.
        """
        im = cv2.imread(im_pth)
        print('Image file is:', im_pth)
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
            print('show_x=%d, im_wt=%d, show_y=%d, im_ht=%d' % (show_x, im_wt, show_y, im_ht))
        scale_x = im_wt*1.0 / show_x
        scale_y = im_ht*1.0 / show_y
        # xml_pth = im_pth.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml').replace('.png', '.xml')
        # We don't assume a standard PASCAL VOC dataset directory. 
        # User should choose image and annotation folder seperately.
        im_head = '.'.join(im_pth.split('/')[-1].split('.')[:-1])
        xml_pth = self.anno_dir.get() + '/' + im_head + '.xml'
        print('XML annotation file is:', xml_pth, end=', ')
        if os.path.exists(xml_pth):
            print('exist')
            boxes = self.parse_xml(xml_pth)
            for box in boxes:
                xmin = int(box.x1/scale_x)
                ymin = int(box.y1/scale_y)
                xmax = int(box.x2/scale_x)
                ymax = int(box.y2/scale_y)
                color = self.get_color_by_cls_name(box.cls_name)
                cv2.rectangle(im, pt1=(xmin, ymin), pt2=(xmax, ymax),
                          color = color, thickness=self.box_thick)
                font_size = 16
                font = ImageFont.truetype('‪C:/Windows/Fonts/msyh.ttc', font_size)
                tx = xmin
                ty = ymin-20
                if(ty<0): 
                    ty = ymin+10
                    tx = xmin+10
                text_org = (tx, ty)
                im = draw_text(im, box.cls_name, text_org, color, font)
        else:
            print("doesn't exist!")
        tkim = self.cv_to_tk(im)
        return tkim

    @staticmethod
    def cv_to_tk(im):
        """Convert OpenCV's (numpy) image to Tkinter-compatible photo image"""
        im = im[:, :, ::-1]  # bgr => rgb
        return ImageTk.PhotoImage(Image.fromarray(im))

    def get_surface_image(self):
        """Return surface image, which is ImageTK type"""
        im = np.ndarray((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                im[y, x, :] = (60, 55, 52) # #34373c(RGB)'s BGR split

        im = cv2.resize(im, ((int)(self.width*0.6), (int)(self.height*0.6)))

        font_size = 30
        font = ImageFont.truetype('‪C:/Windows/Fonts/msyh.ttc', font_size)
        text_org = (self.width*0.16, self.height*0.26)
        text = 'ImageSet Viewer'
        im = draw_text(im, text, text_org, color=(255, 255, 255, 255), font=font)
        
        surface = self.cv_to_tk(im)
        return surface

    def parse_xml(self, xml_pth):
        anno = PascalVOC2007XML(xml_pth)
        boxes = anno.get_boxes()
        return boxes

    def select_image_directory(self):
        im_dir = askdirectory()
        self.listbox.delete(0, len(self.im_names)-1) # delete all elements
        self.fill_im_names(im_dir)

    def select_annotation_directory(self):
        anno_dir = askdirectory()
        self.anno_dir.set(anno_dir) # TODO: validate anno_dir

    def select_save_directory(self):
        save_dir = askdirectory()
        self.save_dir.set(save_dir) # the directory to save(copy) select images

    def fill_im_names(self, im_dir):
        if im_dir is not None:
            self.im_dir.set(im_dir)
            # Get natural order of image file names
            self.im_names = [_ for _ in os.listdir(im_dir)]
            self.im_names.sort()
            for im_name in self.im_names:
                self.listbox.insert(tk.END, im_name)

if __name__ == '__main__':
    # 最简单的方式：不预设im_dir，打开GUI后自行选择图片路径
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
