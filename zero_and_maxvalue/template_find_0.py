import cv2
import numpy as np
# 调整图像的大小并保持高宽比
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 获取图像的大小并初始化尺寸
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)
# 加载模板，转换为灰度，进行canny边缘检测
template = cv2.imread('gauge11.png') 
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)    #转化为灰度值并进行canny边缘检测
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("template", template)

original_image = cv2.imread('gauge01.png')           #将原始图像转化为灰度图
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
found = None
#动态缩放图像以获得更好的模板匹配
for scale in np.linspace(0.1, 3.0, 20)[::-1]:
    resized = maintain_aspect_ratio_resize(gray, width=int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])
    # Stop if template image size is larger than resized image
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break
    # Detect edges in resized image and apply template matching检测调整图片中的边缘并进行模板匹配
    canny = cv2.Canny(resized, 50, 200)#边缘检测
    detected = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF)
    (_, max_val, _, max_loc) = cv2.minMaxLoc(detected)           #模板匹配
    # Uncomment this section for visualization
    '''
    clone = np.dstack([canny, canny, canny])
    cv2.rectangle(clone, (max_loc[0], max_loc[1]), (max_loc[0] + tW, max_loc[1] + tH), (0,255,0), 2)
    cv2.imshow('visualize', clone)
    cv2.waitKey(0)
    '''
    # Keep track of correlation value跟踪相关值，相关值越高，也意味着更高的匹配
    # Higher correlation means better match
    if found is None or max_val > found[0]:
        found = (max_val, max_loc, r)
# Compute coordinates of bounding box
(_, max_loc, r) = found
(start_x, start_y) = (int(max_loc[0] * r), int(max_loc[1] * r))
(end_x, end_y) = (int((max_loc[0] + tW) * r), int((max_loc[1] + tH) * r))
# Draw bounding box on ROI
cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
cv2.imshow('detected', original_image)
cv2.imwrite('detected.png', original_image)
cv2.waitKey(0)
