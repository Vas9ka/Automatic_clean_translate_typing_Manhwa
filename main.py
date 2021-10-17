import io
import base64
import PIL.Image
from cv2 import cv2
from paddleocr import paddleocr
import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cloud_detection
from text_inplacing import find_centers, text_inpaint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def center_distance(x1, x2):
    return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5


def write_result(boxes, txts, eps = 10):
    box_centers = []
    for box in boxes:
        print(box)
        box_centers.append(((box[0][0] + box[2][0]) / 2, (box[0][1] + box[2][1]) / 2))
    distances = []
    for i in range(len(box_centers) - 1):
        for j in range(i + 1, len(box_centers)):
            distances.append(center_distance(box_centers[i], box_centers[j]))
    min_distance = min(distances) + eps
    result = txts[0]
    for i in range(1, len(box_centers)):
        if center_distance(box_centers[i - 1], box_centers[i]) > min_distance:
            result += "\n" + txts[i]
        else:
            result += " " + txts[i]
    return result



def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


if __name__ == "__main__":
    if 'ocr_model' not in st.session_state:
        st.session_state.ocr_model = paddleocr.PaddleOCR(lang='korean', det_db_box_thresh=0.82)
    text_font = ImageFont.truetype('fonts/ko_standard.ttf', 40)
    st.title('Automatic clean and text typing app')
    image_file = st.file_uploader('Upload manhwa page here to get korean text', type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        col1, col2, col3 = st.columns([1, 2, 2])
        image = image_file.read()
        with col1:

            st.image(image)
            image = np.array(Image.open(io.BytesIO(image)).convert('RGB'))
            if 'ocr_result' not in st.session_state or \
                    st.session_state.current_ocr_image.name != image_file.name:
                st.session_state.ocr_result = st.session_state.ocr_model.ocr(image, cls=True)
                st.session_state.current_ocr_image = image_file

        with col2:

            boxes = [line[0] for line in st.session_state.ocr_result]
            txts = [line[1][0] for line in st.session_state.ocr_result]

            for box in boxes:
                box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
                image = cv2.polylines(image, [box], True, (255, 0, 0), 2)
            image = PIL.Image.fromarray(image)
            image_edit = ImageDraw.Draw(image)
            for idx, box in enumerate(boxes):
                text = str(idx + 1)
                image_edit.text((box[0][0] - 25, box[0][1]), text, (255, 0, 0), font=text_font)

            st.image(image)

        with col3:
            for idx, text in enumerate(txts):
                line = str(idx + 1) + '.' + text
                st.text(line)
            if st.button('Download ocr result as a text file'):
                result = write_result(boxes, txts)
                tmp_download_link = download_link(result, 'result.txt', 'Click here to download text!')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
    text = st.file_uploader('Upload text here to put into manhwa', type=['txt'])
    paste_text_image = st.file_uploader('Upload manhwa page here to paste uploaded text', type=['jpg', 'jpeg', 'png'])
    if paste_text_image is not None:
        if 'cloud_det_model' not in st.session_state:
            st.session_state.cloud_det_model = cloud_detection.CloudDetModel()
        col1, col2 = st.columns([1, 1])
        image = paste_text_image.read()
        image = np.array(Image.open(io.BytesIO(image)).convert('RGB'))

        if 'result' not in st.session_state or \
                st.session_state.current_image.name != paste_text_image.name:
            st.session_state.result = st.session_state.cloud_det_model.pred(image)
            st.session_state.current_image = paste_text_image

        with col1:
            st.image(paste_text_image)
        with col2:
            cnts, areas = find_centers(st.session_state.result)
            st.subheader('Size of text:')
            size = st.slider('Choose between 1-30 by 1', min_value=1, max_value=30, step=1)
            image = text_inpaint(image, "Тестируем вставку текста", cnts, areas, 1, size)
            st.image(image)
