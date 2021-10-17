from PIL import Image, ImageFont, ImageDraw
from hyphen import Hyphenator
from hyphen.textwrap2 import fill
import cv2
import imutils
import numpy as np


def find_centers(result):
    clouds_center = {}
    clouds_area = {}
    for i in range(result[0]['masks'].shape[2]):
        image = result[0]['masks'][:, :, i].astype(np.uint8)
        cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            M = cv2.moments(c)
            x, y, w, h = cv2.boundingRect(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            clouds_center[i] = (cX, cY)
            clouds_area[i] = (x, y, w, h)
    return clouds_center, clouds_area


def text_inpaint(image, text, cnts, areas, cnt, size):
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
    x_center = cnts[cnt][0]
    y_center = cnts[cnt][1] - (len(lines) * height / 2)
    height += 2
    if len(lines) > 1:
        if lines[-2][-1] == '-':
            last_word = lines[-2].split(' ')[-1]
            lines[-2] = lines[-2].replace(last_word, '')
            lines[-1] = last_word[:-1] + lines[-1]
    for line in lines:
        if len(line) != 0:
            if line[-1] == "!" or line[-1] == "," or line[-1] == "?":
                last_line = " " + line[-1]
                line[-1] = last_line
    for line in lines:
        width = font.getsize(line)[0]
        d.text((x_center - width / 2, y_center), line, (0, 0, 0), font=font)
        y_center += height
    return image
