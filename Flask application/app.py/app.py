from __future__ import division, print_function
import os
import numpy as np
from keras.preprocessing import image 
from keras.models import load_model
import tensorflow as tf

global graph
graph=tf.get_default_graph()

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("E:\miniproject\Skin_Diseases_1.h5",compile=False)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
           
            print("prediction",preds)
            
        index = ['Acne', 'Melanoma', 'Psoriasis', 'Rosacea', 'Vitiligo']
        
        text = "the predicted skin disease is : " + str(index[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
 
        
    
    
    