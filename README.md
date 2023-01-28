# Final work (Diagnosis of the incidence of Pneumonia II st. according to the results of fluorography)

## Authors (group 15 saturday 28.01.23)
[@termik88](https://github.com/termik88): Илья Колосов

## [Link to StreamLid Cloud](https://termik88-ii-itog-s1-streamlit-app-r1ykkj.streamlit.app/)

## Описание модели

Идея для данной модели была взята с одной из рассматриваемых задачь с Хакатона. ДатаСетом для модели(Нейросетки) служит: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Модель была обучена, сохранена и изъята с [Kaggle](https://www.kaggle.com/)

Итогом работы модели является три показателя, пример:

- Отрицательный анализ: 99.21 %

- Бактериальная пневмония: 0.79 %

- Вирус пневмонии: 0.0 %

Данные показатели являются выходными значениями нейронов сетки.

## Небольщая справка. Чем отличается бактериальная пневмония от вирусной?

При бактериальной пневмонии поражаются именно альвеолы — они воспаляются и отекают, в них скапливается жидкость (экссудат), которая в последующем нагнаивается, мы начинаем кашлять, применяем муколитические средства для разжижения и более легкого отхождения мокроты. Экссудат задерживает прохождение кислорода в кровеносное русло, возникает кислородная недостаточность, снижается сатурация (насыщение крови кислородом). Цель лечения в этом случае — избавление от мокроты путем воздействия на возбудителя, чаще всего пневмококк. Поэтому бактериальная пневмония лечится антибиотиками.


При вирусной пневмонии картина совсем иная. В альвеолах жидкость не скапливается и не нагнаивается. Поэтому кашля с отделяемой мокротой при ковиде нет и быть не может. Вирус поражает стенки сосудов, где происходит газообмен, появляется отек стенок сосудов, сосуды сужаются. Это приводит к замедлению циркуляции крови. Эритроциты «слипаются», препятствуя газообмену, возникает острая нехватка кислорода. Лечить антибиотиками вирусную инфекцию до присоединения к ней бактериальной бесполезно! 