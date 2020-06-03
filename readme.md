# ImageSet Viewer

A GUI, for object detection dataset. Displaying single image with bounding boxes (if any). Now only support PASCAL VOC format.

[![tdozxe.md.png](https://s1.ax1x.com/2020/06/03/tdozxe.md.png)](https://imgchr.com/i/tdozxe)

## Dependencies

- Python

    Support Python 2 & 3

    Anaconda installed Python is recommended

- Python packages

    `pip install image pillow lxml numpy opencv-python`

- apt packages

    `sudo apt-get install python-imaging-tk` # execute this line if pip can't install image

## Supported Features
- Choosing image and annotation directories via button, separately
- Viewing images with annotated bounding boxes via mouse or arrowdown key
- Change box thickness (in code)
- Displaying with specified resizing image height/width (in code)
- Show object class name, supporting Chinese chars
