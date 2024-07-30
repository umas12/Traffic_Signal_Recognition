from flask import Flask
from flask import render_template
from flask import redirect, url_for, request, send_from_directory

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("model.h5")

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)  # Equalize the histogram
    img = img.astype('float32')  # Convert to float32
    img /= 255      # normalizing the img to range from 0 to 1
    return img

def className(classId):
    if classId == 0:
        return "Speed limit (5km/h)"
    elif classId == 1:
        return "Speed limit (15km/h)"
    elif classId == 2:
        return "Speed limit (30km/h)"
    elif classId == 3:
        return "Speed limit (40km/h)"
    elif classId == 4:
        return "Speed limit (50km/h)"
    elif classId == 5:
        return "Speed limit (60km/h)"
    elif classId == 6:
        return "Speed limit (70km/h)"
    elif classId == 7:
        return "speed limit (80km/h)"
    elif classId == 8:
        return "Dont Go straight or left"
    elif classId == 9:
        return "Dont Go straight or Right"
    elif classId == 10:
        return "Dont Go straight"
    elif classId == 11:
        return "Dont Go Left"
    elif classId == 12:
        return "Dont Go Left or Right"
    elif classId == 13:
        return "Dont Go Right"
    elif classId == 14:
        return "Dont overtake from Left"
    elif classId == 15:
        return "No Uturn"
    elif classId == 16:
        return "No Car"
    elif classId == 17:
        return "No horn"
    elif classId == 18:
        return "Speed limit (40km/h)"
    elif classId == 19:
        return "Speed limit (50km/h)"
    elif classId == 20:
        return "Go straight or right"
    elif classId == 21:
        return "Go straight"
    elif classId == 22:
        return "Go Left"
    elif classId == 23:
        return "Go Left or right"
    elif classId == 24:
        return "Go Right"
    elif classId == 25:
        return "keep Left"
    elif classId == 26:
        return "keep Right"
    elif classId == 27:
        return "Roundabout mandatory"
    elif classId == 28:
        return "watch out for cars"
    elif classId == 29:
        return "Horn"
    elif classId == 30:
        return "Bicycles crossing"
    elif classId == 31:
        return "Uturn"
    elif classId == 32:
        return "Road Divider"
    elif classId == 33:
        return "Traffic signals"
    elif classId == 34:
        return "Danger Ahead"
    elif classId == 35:
        return "Zebra Crossing"
    elif classId == 36:
        return "Bicycles crossing"
    elif classId == 37:
        return "Children crossing"
    elif classId == 38:
        return "Dangerous curve to the left"
    elif classId == 39:
        return "Dangerous curve to the right"
    elif classId == 40:
        return "Unknown1"
    elif classId == 41:
        return "Unknown2"
    elif classId == 42:
        return "Unknown3"
    elif classId == 43:
        return "Go right or straight"
    elif classId == 44:
        return "Go left or straight"
    elif classId == 45:
        return "Unknown4"
    elif classId == 46:
        return "ZigZag Curve"
    elif classId == 47:
        return "Train Crossing"
    elif classId == 48:
        return "Under Construction"
    elif classId == 49:
        return "Unknown5"
    elif classId == 50:
        return "Fences"
    elif classId == 51:
        return "Heavy Vehicle Accidents"
    elif classId == 52:
        return "Unknown6"
    elif classId == 53:
        return "Give Way"
    elif classId == 54:
        return "No stopping"
    elif classId == 55:
        return "No entry"
    elif classId == 56:
        return "Unknown7"
    elif classId == 57:
        return "Unknown8"
    
def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(32, 32))
    img = np.asarray(img)
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)     # 1 image prediction at a time of size 32x32 in grayscale
    # PREDICT IMAGE
    predictions = model.predict(img)
    class_index =np.argmax(predictions, axis=1)[0]
    preds = className(class_index)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    if f and f.filename != '':
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'test_data', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        file_url = url_for('uploaded_file', filename=f.filename)
        return render_template('index.html', result=result, file_url=file_url)
    else:
        return render_template('index.html', result="No file selected")
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('test_data', filename)

if __name__ == '__main__':
    app.run(port=8000, debug=True)