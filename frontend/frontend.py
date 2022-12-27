import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import requests
import io
from PIL import Image
import json
import cv2
import numpy


app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, 'static')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_URL = 'http://backend:5555/predict'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #image_path = None
    variable = ' '
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', variable = variabl, var2='no image')

        file = request.files['file']
       
        if file.filename == '':
            return render_template('index.html', variable = variable, var2='no image')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            variable = filename

            img = Image.open(file.stream)

            img = img.convert('RGB')
            open_cv_image = numpy.array(img)
            # Convert RGB to BGR 
            img = open_cv_image[:, :, ::-1].copy()

            # encode image as jpeg
            _, img_encoded = cv2.imencode('.jpg', img)
            # send http request with image and receive response
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}

            response = requests.post(MODEL_URL, data=img_encoded.tostring(), headers=headers)

            json_response = response.json()

            if bool(json_response.get('Successful')):
                text = json_response['Text']
                return render_template('index.html', variable = variable, var2 = text)
    
    return render_template('index.html', variable = variable, var2 = 'no image')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8899)

