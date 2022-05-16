import io
import os
import sys

import PIL.Image
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from cv2 import cv2
from paddleocr import paddleocr
from streamlit_drawable_canvas import st_canvas
import translate

INPAINT_DIR = os.path.abspath('inpainting/')
sys.path.append(INPAINT_DIR)
from inpainting import inpaint

import cloud_detection
from text_inplacing import find_centers, text_inpaint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
custom_config = r'--oem 1 --psm 3'


def center_distance(x1, x2):
    return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5


def write_result(boxes, txts, eps=10):
    box_centers = []
    for box in boxes:
        print(box)
        box_centers.append(((box[0][0] + box[2][0]) / 2, (box[0][1] + box[2][1]) / 2))
    if len(txts) == 1:
        distances = [0]
    else:
        distances = []
    for i in range(len(box_centers) - 1):
        for j in range(i + 1, len(box_centers)):
            distances.append(center_distance(box_centers[i], box_centers[j]))
    try:
        min_distance = min(distances) + eps
        result = txts[0]
        for i in range(1, len(box_centers)):
            if center_distance(box_centers[i - 1], box_centers[i]) > min_distance:
                result += "\n" + txts[i]
            else:
                result += " " + txts[i]
        return result.split('\n')
    except:
        return None


if __name__ == "__main__":
    st.title('Automatic Manhwa localization app')

    text_font = ImageFont.truetype('fonts/ko_standard.ttf', 40)
    if 'ocr_model' not in st.session_state:
        st.session_state.ocr_model = paddleocr.PaddleOCR(lang='korean', det_db_box_thresh=0.50,
                                                         det_model_dir='uptraining')
    if 'cloud_det_model' not in st.session_state:
        st.session_state.cloud_det_model = cloud_detection.CloudDetModel()

    image_file = st.file_uploader('Upload manhwa page here', type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        image = image_file.read()
        st.image(image)
        image = np.array(Image.open(io.BytesIO(image)).convert('RGB'))
        image_inpaint = image.copy()
        init_image = image.copy()
        mask = np.zeros((image.shape[0], image.shape[1]))

        if 'ocr_result' not in st.session_state or \
                st.session_state.current_ocr_image.name != image_file.name:
            st.session_state.ocr_result = st.session_state.ocr_model.ocr(image, cls=True)
            st.session_state.current_ocr_image = image_file
        if 'result' not in st.session_state or \
                'current_cloud_image' not in st.session_state or \
                st.session_state.current_cloud_image.name != image_file.name:
            st.session_state.result = st.session_state.cloud_det_model.pred(image)
            st.session_state.current_cloud_image = image_file
        boxes = [line[0] for line in st.session_state.ocr_result]
        txts = [line[1][0] for line in st.session_state.ocr_result]
        for box in boxes:
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            rect = cv2.boundingRect(np.asarray(box).astype('int'))
            x, y, w, h = rect
            x -= 5
            y -= 5
            w += 10
            h += 10
            text = cv2.Canny(image[y: y + h, x: x + w], 200, 500)
            mask[y: y + h, x: x + w] = text
            image = cv2.polylines(image, [box], True, (255, 0, 0), 2)
        resized_image = Image.fromarray(image).copy()
        resized_image.thumbnail((Image.fromarray(image).height / 1.25, Image.fromarray(image).width / 1.25))
        canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=1,
                stroke_color="rgba(255,0,0)",
                background_color="rgba(255,255,255)",
                background_image=resized_image,
                update_streamlit=True,
                height=resized_image.height,
                width=resized_image.width,
                drawing_mode="rect",
                point_display_radius=0,
                key=image_file.name,
        )
        try:
            result = canvas_result.json_data['objects']
            x_scale = Image.fromarray(image).width / resized_image.width
            y_scale = Image.fromarray(image).height / resized_image.height
            for i in range(len(result)):
                x, y, w, h = round(result[i]['left'] * x_scale), round(result[i]['top'] * y_scale), round(
                result[i]['width'] * x_scale), round(result[i]['height'] * y_scale)
                text = cv2.Canny(np.asarray(image)[y: y + h, x: x + w], 200, 500)
                mask[y: y + h, x: x + w] = text
        except:
            pass
        mask = np.array(mask, dtype=np.uint8)
        mask = cv2.dilate(np.asarray(mask),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image = PIL.Image.fromarray(image)
        image_edit = ImageDraw.Draw(image)
        for idx, box in enumerate(boxes):
            text = str(idx + 1)
            image_edit.text((box[0][0] - 25, box[0][1]), text, (255, 0, 0), font=text_font)
        if st.checkbox('All text selected', key=image_file.name + 'box'):
            image = np.asarray(image)
            result = write_result(boxes, txts)
            print(result)

            if 'inpainted_image' not in st.session_state or \
                'inpainted_result' not in st.session_state or \
                st.session_state.inpainted_image.name != image_file.name:
                image_inpaint = inpaint.inpaint(image_inpaint, mask)
                st.session_state.inpainted_image = image_file
                st.session_state.inpainted_result = image_inpaint
            cnts, areas = find_centers(st.session_state.result)
            sizes = np.zeros(len(cnts))
            try:
                texts = translate.translate(result)
            except:
                st.write("Seems there is no text on the image")
                texts = []

            example_txts = []
            for i in range(len(texts)):
                example_txts.append(texts[i]['text'])
            inpainted_image = st.session_state.inpainted_result.copy()
            print(example_txts)
            st.image(inpainted_image)
            col1, col2, col3 = st.columns([1, 2, 3])
            with col1:
                print(len(areas))
                for i in range(len(example_txts)):
                    text = example_txts[i]
                    text_size = 10
                    length = len(text)
                    while length > 20:
                        length /= 2
                    while text_size * length <= areas[i][2]:
                        text_size += 1
                    st.subheader(f'Size of {i + 1} cloud text:')
                    size = st.slider('Choose between 1-100 by 1', min_value=1, max_value=100, step=1,
                                            value=text_size,
                                             key=i)
                    sizes[i] = size
            with col2:
                for i in range(len(example_txts)):
                    txts[i] = st.text_input(value=example_txts[i], label=f'Text in {i + 1} cloud')
            with col3:
                for i in range(len(example_txts)):
                    inpainted_image = text_inpaint(inpainted_image, txts[i], cnts, areas, i, int(sizes[i]))
                st.image(inpainted_image)
