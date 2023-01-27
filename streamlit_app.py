import io
import streamlit as st
from PIL import Image

from tensorflow import keras
import numpy as np
import cv2 as cv


@st.cache(allow_output_mutation=True)
def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_model():
    return keras.models.load_model('model_v2')


def preprocessor(data):
    temp = []
    img = cv.imread(data)
    if type(img) == np.ndarray:
        img1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img2 = cv.blur(img1,(3,3))
        temp.append(cv.resize(img2,(32,32)))
        scaled = np.array(temp)/255
    return scaled.reshape(scaled.shape[0], scaled.shape[1], scaled.shape[2], 1)


def get_answer(predict):
    def get_aswer(value, predict):
        return value + ' Predict: ' + str(max(predict[0])*100) + "%"

    pred = np.argmax(predict)
    
    if pred == 0: return get_aswer('Bacterial pneumonia', predict)
    elif pred == 1: return get_aswer('Normal', predict)
    elif pred == 2: return get_aswer('Bitus pnrumonia', predict)


model = load_model()


st.title('Диагностика заболеваемости Пневмонией II ст. по результатам флюрографии ')
img = load_image()
result = st.button('Распознать изображение')
if result:
    img_scaler = preprocessor(img)
    pred = model.predict(img_scaler)
    st.write('**Результаты распознавания:**')
    st.write(get_answer(pred))
