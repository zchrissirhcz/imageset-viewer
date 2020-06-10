# ImageSet Viewer

[ImageSet Viewer](https://github.com/zchrissirhcz/imageset-viewer) is a GUI program, visualizing labeled boxes and categories for PASCAL VOC format.

Support: Python 2 & 3, Windows / Linux / MacOSX desktop.

_News_:
> 2020-06-09 23:29:21
> Support picking up images by left control

## Install

```bash
git clone https://github.com/zchrissirhcz/imageset-viewer
cd imageset-viewer
pip install -r requirements.txt
```


## Usage

**Open GUI**
```bash
python imageset-viewer.py
```


![screenshot](https://user-images.githubusercontent.com/3831847/84168090-94bf9580-aaa9-11ea-9aeb-a56d476e2610.png)



**View Labels**

Choose image and annotation(xml) directories first. 

Use $\uparrow$ and $\downarrow$ keys, mouse left click and drag scoll bar for navigation.


**Pick up images**

Choose saving directory first.

Press left control key(`Control_L`) to save(copy) current image.


**Zoom out big image**

Specify `show_x` and `show_y` in code.


## License

MIT
