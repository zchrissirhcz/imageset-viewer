# ImageSet Viewer

[ImageSet Viewer](https://github.com/zchrissirhcz/imageset-viewer) is a GUI program, visualizing labeled boxes and categories for Pascal VOC / ImageNet object detection format.

Support: Python 2 & 3, Windows / Linux / MacOSX desktop.

_News_:
> 2020-06-16 11:39  
> Support specifying ignore and not ignore class names (in code)


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

Use &#8593;(up arrow) and &#8595;(down arrow) keys, mouse left click and drag scoll bar for navigation.


**Pick up images**

Choose saving directory first.

Press left control key(`Control_L`) to save(copy) current image.


**Zoom out big image**

Specify `max_width` and `max_height` in code.


**Show customized VOC-format-labeled dataset**

Check `example3()` function in [imageset-viewer.py](imageset-viewer.py) for imitation.


**Show remote server's imageset**

You have multiple choices:

- In remote Linux server, setup samba then use addresses like `\\172.15.12.34\some\dir` (Recommended)

- In remote Linux server and local machine, setup X11 forwarding,  then run `imageset-viewer.py` in server and show in local machine.
    - PyCharm professional
    - Win10: [Xming](https://sourceforge.net/projects/xming/) + [Putty](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)


## License

[MIT](LICENSE)
