from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import load_model
import os
import sys
import numpy as np
from keras_preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import sys
import os
import glob
import re

app = Flask(__name__)

model = load_model(r'C:\Users\Hardik\Desktop\envflask\Houseimgclassify\house1.h5')
model.make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size = (200,200))

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis =0)

    x = preprocess_input(x,mode = 'caffe')

    preds = model.predict(x)

    return preds

@app.route('/', methods= ['GET'])
def home():
    return "Hello"

@app.route('/predict', methods = ['GET','POST'])

def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads',secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path,model)

        # pred_class = decode_predictions(preds)
        result = preds
        print(result[0][0],"aaaa")

        if int(result[0][0])==1:
            a = "house exterior"
            return render_template("predict.html",result = a)
        else:
            b = "house interior"
            return render_template("predict.html",result = b)
    return render_template("predict.html")
  
if __name__ == ('__main__'):
    app.run(debug = True)

