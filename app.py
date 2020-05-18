from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import time
#Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


import json

import base64

#Define a flask app
app = Flask(__name__)
@app.route('/')
def index_view():
    return render_template('index.html')
@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/results')
def result():
    return render_template('results.html')



#parameters

args={'prototxt':'models/colorization_deploy_v2.prototxt','model':'models/colorization_release_v2.caffemodel','points':'models/pts_in_hull.npy'}
#Model loading...
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
pts = np.load(args["points"])
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]  

print('Model loaded. Start serving...')


def model_predict(img_path):
        # load the input image from disk, scale the pixel intensities to the
        # range [0, 1], and then convert the image from the BGR to Lab color
        # space
        image = cv2.imread(img_path)
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # resize the Lab image to 224x224 (the dimensions the colorization
        # network accepts), split channels, extract the 'L' channel, and then
        # perform mean centering
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # pass the L channel through the network which will *predict* the 'a'
        # and 'b' channel values
        'print("[INFO] colorizing image...")'
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        # resize the predicted 'ab' volume to the same dimensions as our
        # input image
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # grab the 'L' channel from the *original* input image (not the
        # resized one) and concatenate the original 'L' channel with the
        # predicted 'ab' channels
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # convert the output image from the Lab color space to RGB, then
        # clip any values that fall outside the range [0, 1]
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)

        # the current colorized image is represented as a floating point
        # data type in the range [0, 1] -- let's convert to an unsigned
        # 8-bit integer representation in the range [0, 255]
        colorized = (255 * colorized).astype("uint8")
        return colorized



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        for filename in os.listdir('static/'):
            if filename.startswith('gg'): 
                os.remove('static/' + filename)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        x="gguploaded"+str(time.time())+".jpg"
        file_path = os.path.join(
            basepath, 'static', secure_filename(x))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        x1="ggcolorized"+str(time.time())+".jpg"
        cv2.imwrite("static/"+x1, preds)

        return render_template("results.html", graph=x,fileName=x1)

    return None


if __name__ == '__main__':
    #app.run(debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
