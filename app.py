# -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__, static_folder='static')

# ����ģ�ͺ�����б�
model_path = "H:/Ceshi/Food/models/model.h5"
classes_path = "H:/Ceshi/Food/classes.txt"
model = keras.models.load_model(model_path)
with open(classes_path) as f:
    classes = f.read().splitlines()
# -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__, static_folder='static')

# �����ϴ��ļ�Ŀ¼
upload_folder = 'H:/Ceshi/Food/static'
app.config['UPLOAD_FOLDER'] = upload_folder

# ����ģ�ͺ�����б�
model_path = "H:/Ceshi/Food/models/model.h5"
classes_path = "H:/Ceshi/Food/classes.txt"
model = keras.models.load_model(model_path)
with open(classes_path) as f:
    classes = f.read().splitlines()


@app.route('/', methods=['GET', 'POST'])
def classify_food():
    if request.method == 'POST':
        # �ϴ���Ԥ��ͼƬ
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        img = load_img(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), target_size=(224, 224))
        img_array = img_to_array(img)
        processed_img = preprocess_img(img_array)
        processed_img = np.expand_dims(processed_img, axis=0)

        prediction = model.predict(processed_img)
        predicted_class = classes[np.argmax(prediction)]

        return render_template('result.html', image_file=file.filename, predicted_class=predicted_class)
    else:
        return render_template('index.html')


# ͼƬԤ����        
def preprocess_img(img):       
    # ���Ԥ�������
    # �������滻Ϊ�ʵ���ͼ��Ԥ�����߼�
    processed_img = img  # ʾ�����������κ�Ԥ����ֱ�ӷ���ԭʼͼ��
    return processed_img


if __name__ == '__main__':
    app.run(debug=False)