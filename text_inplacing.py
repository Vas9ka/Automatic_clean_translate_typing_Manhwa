from PIL import Image, ImageFont, ImageDraw
from hyphen import Hyphenator
from hyphen.textwrap2 import fill
import cv2
import imutils
import numpy as np
import operator

def find_centers(result):
    clouds_center = []
    clouds_area = []
    for i in range(result[0]['masks'].shape[2]):
        image = result[0]['masks'][:, :, i].astype(np.uint8)
        cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            M = cv2.moments(c)
            x, y, w, h = cv2.boundingRect(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            clouds_center.append((cX, cY))
            clouds_area.append((x, y, w, h))
        clouds_center.sort(key=operator.itemgetter(1))
        clouds_area.sort(key=operator.itemgetter(1))
    return clouds_center, clouds_area


def text_inpaint(image, text, cnts, areas, cnt, size, color=(0, 0, 0)):
    if not text:
        return np.asarray(image)
    width = 10
    image = Image.fromarray(image)
    font = ImageFont.truetype('fonts/anime-ace-v05.ttf', size)
    d = ImageDraw.Draw(image)
    h_ru = Hyphenator('ru_RU')
    wrapper = fill(text, width, use_hyphenator=h_ru)
    lines = wrapper.split('\n')
    height = font.getsize(lines[0])[1]
    while width * size > areas[cnt][2]:
        width -= 1
    while width * size < areas[cnt][2]:
        width += 1
    text = fill(text, width, use_hyphenator=h_ru)
    lines = text.split('\n')
    print(lines)
    print(width)
    x_center = cnts[cnt][0]
    y_center = cnts[cnt][1] - (len(lines) * height / 2)
    height += 2
    right_lines = []
    for line in lines:
        if '-' in line[-1]:
            right_lines.append(line.split('-')[0] + '-')
        else:
            right_lines.append(line)
    for line in lines:
        width = font.getsize(line)[0]
        d.text((x_center - width / 2, y_center), line, color, font=font)
        y_center += height
    return np.asarray(image)
