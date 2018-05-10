# ImageSet Viewer

A GUI, for object detection dataset. Displaying single image with bounding boxes (if any). Now only support PASCAL VOC format.

![](./screenshot.png)

## Dependencies
- OpenCV's python interface, e.g. cv2
- `pip install --upgrade image pillow lxml numpy`

## Supported Function
- Choosing image folder via button
- Viewing images with annotated bounding boxes via mouse or arrowdown key
- Change box thick (in code)
- Displaying with specified resizing image height/width (in code)