# ImageSet Viewer

[ImageSet Viewer](https://github.com/zchrissirhcz/imageset-viewer) is a Python Tkinter based GUI program, displaying images with corresponding labels. Currently it support PASCAL VOC object detection format, the very popular xml-format-based bbox annotation format, drawing bounding boxes and class names for annotated objects. For people who do object detection training, concerns what labeled objects are looks like, concerns if objects are labeled with correct categories, and don't want
rewrite "show-boxes" programs over and over again, just pick ImageSet Viewer and use it!


## Getting started

Install:
```bash
git clone https://github.com/zchrissirhcz/imageset-viewer
cd imageset-viewer
pip install -r requirements.txt
```

Run:
```bash
python imageset-viewer.py
```
Choosing image and annotation directory seperately, you'll see annotated images. Use mouse or arrow keys to switch to different images.

[![tdozxe.md.png](https://s1.ax1x.com/2020/06/03/tdozxe.md.png)](https://imgchr.com/i/tdozxe)


## License

MIT
