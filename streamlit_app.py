import streamlit as st
from tensorflow import keras
import numpy as np
#для CV2 необходима библиотека opencv-python
#но для облака Streamlid она не подходит, будет ошибка
#Поэтому необходимо устанавливать opencv-python-headless 
import cv2 as cv


@st.cache(allow_output_mutation=True)
def load_model():
    #модель была создана из работы
    #https://www.kaggle.com/code/vijay20213/pneumonia-detection-with-cnn-and-ml-with-98-acc/notebook
    return keras.models.load_model('model_v2')


def preprocessor(img_file_bytes):
    #cv.inread работает с загружаемыми файлами
    #cv.imdecode с файлами из памяти, но предварительно конвертированные
    temp = []
    opencv_image = cv.imdecode(img_file_bytes, 1)
    if type(opencv_image) == np.ndarray:
        opencv_image1 = cv.cvtColor(opencv_image,cv.COLOR_BGR2GRAY)
        opencv_image2 = cv.blur(opencv_image1,(3,3))
        temp.append(cv.resize(opencv_image2,(32,32)))
        scaled = np.array(temp)/255
        tenser_img = scaled.reshape(scaled.shape[0], scaled.shape[1], scaled.shape[2], 1)
    return tenser_img


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        #отобразить загруженную картинку на странице
        st.image(uploaded_file.getvalue())
        #Конвертация файла
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return file_bytes
    else:
        return None


def get_dict_aswers(predict):
    def get_aswer(predict):
        return round(predict*100, 2)

    dict = {}
    
    dict['Бактериальная пневмония: '] = get_aswer(predict[0][0])
    dict['Отрицательный анализ: '] = get_aswer(predict[0][1])
    dict['Вирус пневмонии: '] = get_aswer(predict[0][2])
    return dict


def get_answer(dict_answers):
    #сортируем в обратном порядке
    dict_answers_sort = sorted(dict_answers.items(), key=lambda x:x[1], reverse=True)
    
    for i in range(len(dict_answers_sort)):
        st.write(dict_answers_sort[i][0], dict_answers_sort[i][1], '%')


model = load_model()


st.title('Диагностика заболеваемости Пневмонией II ст. по результатам флюрографии ')
img_file_bytes = load_image()
result = st.button('Распознать изображение')
if result:
    img_scaler = preprocessor(img_file_bytes)
    predict = model.predict(img_scaler)
    st.write('**Вероятность диагноза:**')
    dict_answers = get_dict_aswers(predict)
    get_answer(dict_answers)
