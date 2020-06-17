#!/usr/bin/env python
#coding: utf-8

# coding: utf-8

"""
Demonstrates how to resizing tkinter windows and image on it

https://cloud.tencent.com/developer/article/1430998

https://stackoverflow.com/questions/7299955/tkinter-binding-a-function-with-arguments-to-a-widget
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("800x600")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.lbPic = tk.Label(self, text='test', compound='center')

        self.im_orig = cv2.imread('/Users/chris/data/VOC2007/JPEGImages/000001.jpg')

        self.xmin_orig = 8
        self.ymin_orig = 12
        self.xmax_orig = 352
        self.ymax_orig = 498

        cv2.rectangle(
            self.im_orig,
            pt1 = (self.xmin_orig, self.ymin_orig),
            pt2 = (self.xmax_orig, self.ymax_orig),
            color = (0, 255, 0),
            thickness = 2
        )

        self.im_orig = self.im_orig[:, :, ::-1]  # bgr => rgb   necessary

        tkim = ImageTk.PhotoImage(Image.fromarray(self.im_orig))
        self.lbPic['image'] = tkim
        self.lbPic.image = tkim

        self.lbPic.bind('<Configure>', self.changeSize)
        self.lbPic.grid(row=0, column=0, sticky=tk.NSEW)

    def changeSize(self, event):
        im = cv2.resize(self.im_orig, (event.width, event.height))

        tkim = ImageTk.PhotoImage(Image.fromarray(im))
        self.lbPic['image'] = tkim
        self.lbPic.image = tkim


def main():
    app = App()
    app.title('缩放图像')

    app.mainloop()


if __name__ == '__main__':
    main()
