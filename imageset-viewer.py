#!/usr/bin/env python
# coding: utf-8

__author__ = 'Zhuo Zhang'
__copyright__ = 'Copyright 2017-2020, Zhuo Zhang'
__license__ = 'MIT'
__version__ = '0.4'
__email__ = 'imzhuo@foxmail.com'
__status__ = 'Development'
__description__ = 'Tkinter based GUI, visualizing PASCAL VOC object detection annotation'

"""
Changelog:

- 2020-06-13 00:48   v0.4
    API change: add class name mapping dict, mapping xml class name to shown class name.
    Based on this, ImageNet2012 and self-defined VOC format style dataset labels can show.
    Supported image extension: bmp, jpg, jpeg, png and their upper cases.

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
import platform
import matplotlib.font_manager as fm # to create font
import six
import logging
from natsort import natsorted

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

if six.PY3:
    import tkinter as tk
    from tkinter.filedialog import askdirectory
else:
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
    return np.array(im_pil)


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
    hsv_tuples = [(x*1.0 / num_cls, 1., 1.) for x in range(num_cls)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(42)
    random.shuffle(colors)
    random.seed(None)
    return colors


class VOC_Viewer(tk.Tk):
    def __init__(self, im_dir=None, anno_dir=None, save_dir=None, max_width=None, max_height=None, box_thick=1, cls_name_to_show=None):
        # 加载图像：tk不支持直接使用jpg图片。需要Pillow模块进行中转
        """
        @param im_dir: 包含图片的路径，也就是"JPEGImages". 要求它的同级目录中包含Annotations目录，里面包含各种xml文件。
        @param show_width: 图片显示时候的最大宽度
        @param show_height: 图片显示时的最大高度
        @param box_thick: thickness of bounding box
        """
        #super().__init__()
        tk.Tk.__init__(self)

        # custom settings
        self.max_width = max_width
        self.max_height = max_height
        self.box_thick = box_thick
        self.bg = '#34373c'
        self.fg = '#f2f2f2'
        # MacOSX's tk is wired and I don't want tkmacosx
        if platform.system()=='Darwin':
            self.bg, self.fg = self.fg, self.bg

        # set title, window size and background
        self.title('ImageSet Viewer ' + __version__)
        self.width = (int)(0.6 * self.winfo_screenwidth())
        self.height = (int)(0.6 * self.winfo_screenheight())
        self.geometry('%dx%d+200+100' % (self.width, self.height))
        self.configure(bg=self.bg)
        self.minsize(800, 600)

        self.init_components(im_dir, anno_dir, save_dir)
        self.init_dataset(cls_name_to_show)

    def init_dataset(self, cls_name_to_show):
        if cls_name_to_show is None:
            self.cls_names = [ #'__background__',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
            ]
            self.cls_name_to_show = dict()
            for item in self.cls_names:
                self.cls_name_to_show[item] = item
        else:
            self.cls_name_to_show = cls_name_to_show
            self.cls_names = [_ for _ in cls_name_to_show.keys()]

        self.num_classes = len(self.cls_names)
        self.color_table = get_color_table(self.num_classes)
        self.class_to_ind = dict(zip(self.cls_names, range(self.num_classes)))
        self.supported_im_ext = ['bmp', 'png', 'jpg', 'jpeg', 
                                 'BMP', 'PNG', 'JPG', 'JPEG']

    def get_color_by_cls_name(self, cls_name):
        ind = self.class_to_ind[cls_name]
        return self.color_table[ind]

    def init_components(self, im_dir, anno_dir, save_dir):
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
            self.im_names = natsorted(self.im_names)
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

        if anno_dir is not None:
            self.anno_dir.set(anno_dir)

        # copy (save) dir button
        choose_save_dir_btn = tk.Button(directory_frame, text='Copy Save Directory',
            command=self.select_save_directory, bg=self.bg, fg=self.fg)
        choose_save_dir_btn.grid(row=2, column=0, sticky=tk.NSEW)

        self.save_dir = tk.StringVar()
        save_dir_entry = tk.Entry(directory_frame, text=self.save_dir, state='readonly')
        save_dir_entry.grid(row=2, column=1, sticky=tk.NSEW)

        if save_dir is not None:
            self.save_dir.set(save_dir)

    def callback(self, event=None):
        im_id = self.listbox.curselection()
        if im_id:
            im_id = im_id[0]
            logging.info('im_id is {:d}'.format(im_id))
            im_name = self.listbox.get(im_id)
            im_ext = im_name.split('.')[-1]
            if im_ext in self.supported_im_ext:
                im_pth = os.path.join(self.im_dir.get(), im_name).replace('\\', '/')
                self.tkim = self.get_tkim(im_pth)
                self.image_label.configure(image=self.tkim)
                #logging.debug(im_pth)

    def save_image(self, event):
        """保存（拷贝）选中的图片到目录
        当前设定为，按左Control键，把当前浏览的图片存储到指定的保存路径。用于手工挑选图片
        """
        im_id = self.listbox.curselection()
        if im_id:
            im_name = self.listbox.get(im_id)
            im_ext = im_name.split('.')[-1]
            if im_ext in self.supported_im_ext:
                im_pth = os.path.join(self.im_dir.get(), im_name).replace('\\', '/')
                save_pth = os.path.join(self.save_dir.get(), im_name).replace('\\', '/')
                shutil.copyfile(im_pth, save_pth)
                logging.info('Save(copy) to {:s}'.format(save_pth))
                #logging.debug(im_pth)

    def get_tkim(self, im_pth):
        """
        Load image and annotation, draw on image, and convert to image.
        When necessary, image resizing is utilized.
        """
        im = cv2.imread(im_pth)
        logging.info('Image file is: {:s}'.format(im_pth))
        im_ht, im_wt, im_dt = im.shape
        if self.max_width is None or self.max_width >= im_wt:
            show_width = im_wt
        else:
            show_width = self.max_width

        if self.max_height is None or self.max_height >= im_ht:
            show_height = im_ht
        else:
            show_height = self.max_height

        scale_width = im_wt * 1.0 / show_width
        scale_height = im_ht * 1.0 / show_height

        if show_width!=im_wt or show_height!=im_ht:
            im = cv2.resize(im, (show_width, show_height))
            logging.info('doing resize, show_width={:d}, im_wt={:d}, show_height={:d}, im_ht={:d}'.format(show_width, im_wt, show_height, im_ht))
        
        # xml_pth = im_pth.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml').replace('.png', '.xml')
        # We don't assume a standard PASCAL VOC dataset directory.
        # User should choose image and annotation folder seperately.
        im_head = '.'.join(im_pth.split('/')[-1].split('.')[:-1])
        xml_pth = self.anno_dir.get() + '/' + im_head + '.xml'
        if os.path.exists(xml_pth):
            logging.info('XML annotation file is {:s}'.format(xml_pth))
            boxes = self.parse_xml(xml_pth)
            for box in boxes:
                if (self.class_to_ind.get(box.cls_name, -1)==-1):
                    # The class name parsed from XML not in specified class names, ignore it
                    # continue
                    pass
                xmin = int(box.x1/scale_width)
                ymin = int(box.y1/scale_height)
                xmax = int(box.x2/scale_width)
                ymax = int(box.y2/scale_height)
                color = self.get_color_by_cls_name(box.cls_name)
                cv2.rectangle(im, pt1=(xmin, ymin), pt2=(xmax, ymax),
                          color = color, thickness=self.box_thick)
                font_size = 16
                font = self.get_font(font_size)
                tx = xmin
                ty = ymin-20
                if(ty<0):
                    ty = ymin+10
                    tx = xmin+10
                text_org = (tx, ty)
                show_text = self.cls_name_to_show[box.cls_name]
                logging.debug('box.cls_name is:' + box.cls_name)
                logging.debug('show_text:' + show_text)
                im = draw_text(im, show_text, text_org, color, font)
        else:
            logging.warn("XML annotation file {:s} doesn't exist".format(xml_pth))
        return self.cv_to_tk(im)

    @staticmethod
    def cv_to_tk(im):
        """Convert OpenCV's (numpy) image to Tkinter-compatible photo image"""
        im = im[:, :, ::-1]  # bgr => rgb
        return ImageTk.PhotoImage(Image.fromarray(im))

    @staticmethod
    def get_font(font_size):
        font_pth = None
        if platform.system()=='Windows':
            font_pth = 'C:/Windows/Fonts/msyh.ttc'
        elif (platform.system()=='Linux'):
            font_pth = fm.findfont(fm.FontProperties(family='DejaVu Mono'))
        else:
            font_pth = 'Helvetica'
        return ImageFont.truetype(font_pth, font_size)

    def get_surface_image(self):
        """Return surface image, which is ImageTK type"""
        im = np.ndarray((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                im[y, x, :] = (60, 55, 52) # #34373c(RGB)'s BGR split

        im = cv2.resize(im, ((int)(self.width*0.6), (int)(self.height*0.6)))

        font_size = 30
        font = self.get_font(font_size)
        text_org = (self.width*0.16, self.height*0.26)
        text = 'ImageSet Viewer'
        im = draw_text(im, text, text_org, color=(255, 255, 255, 255), font=font)

        return self.cv_to_tk(im)

    def parse_xml(self, xml_pth):
        anno = PascalVOC2007XML(xml_pth)
        return anno.get_boxes()

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
            self.im_names = natsorted(self.im_names)
            for im_name in self.im_names:
                self.listbox.insert(tk.END, im_name)

def example1():
    """例子1：最简单模式：什么参数都不指定，在GUI中手动选择图片和XML路径"""
    app = VOC_Viewer()
    app.mainloop()

def example2():
    """例子2：分别指定各种参数。以标准PASCAL VOC为例"""
    # 类别字典，key是xml中所写类别，value是希望在GUI界面上显示的类别
    voc_cls_dict = {
        '__background__': '背景',
        'aeroplane': '飞机',
        'bicycle': '自行车',
        'bird': '鸟',
        'boat': '船',
        'bottle': '瓶子',
        'bus': '公交车',
        'car': '汽车',
        'cat': '猫',
        'chair': '椅子',
        'cow': '牛',
        'diningtable': '餐桌',
        'dog': '狗',
        'horse': '马',
        'motorbike': '摩托车',
        'person': '人',
        'pottedplant': '盆栽',
        'sheep': '绵羊',
        'sofa': '沙发',
        'train': '火车',
        'tvmonitor': '显示器'
    }
    app = VOC_Viewer(im_dir = 'D:/data/VOC2007/JPEGImages',   # 图片目录
                    anno_dir = 'D:/data/VOC2007/Annotations', # xml目录
                    save_dir = 'D:/data/VOC2007/save',  # 挑图保存目录
                    max_width = 1000,   # 显示图片宽度做多1000像素
                    max_height = 800,   # 显示图片高度最多1000像素
                    box_thick = 2,   # bbox边框宽度
                    cls_name_to_show = voc_cls_dict
                    )
    app.mainloop()

def example3():
    """例子3：分别指定各种参数, ImageNet2012"""

    fin = open('imagenet_cls_cn.txt', encoding='UTF-8')
    lines = [_.strip() for _ in fin.readlines()]
    fin.close()

    ilsvrc2012_cls_dict = dict()
    for item in lines:
        item = item.split(' ')
        digit_cls_name = item[0]
        literal_cls_name = ' '.join(item[1:])
        ilsvrc2012_cls_dict[digit_cls_name] = literal_cls_name

    app = VOC_Viewer(im_dir = 'D:/data/ILSVRC2012/ILSVRC2012_img_train/n01440764',   # 图片目录
                    anno_dir = 'D:/data/ILSVRC2012/ILSVRC2012_bbox_train_v2/n01440764', # xml目录
                    save_dir = None,  # 挑图保存目录
                    max_width = 1000,   # 显示图片宽度做多1000像素
                    max_height = 800,   # 显示图片高度最多1000像素
                    box_thick = 2,  # bbox边框宽度
                    cls_name_to_show = ilsvrc2012_cls_dict
                    )
    app.mainloop()



if __name__ == '__main__':
    example1()
    #example2()
    #example3()