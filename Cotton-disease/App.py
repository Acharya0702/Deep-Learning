from flask import Flask, request, render_template
from keras.utils import load_img
from keras.models import load_model
import numpy as np
import os
model=load_model('effnet.h5')
app=Flask(__name__)
imgFolder=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=imgFolder
@app.route('/')
def home():
	return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    x=''
    imagefile=request.files['f']
    image_path='static/images/'+imagefile.filename
    imagefile.save(image_path)
    print(image_path)
    image=load_img(image_path, target_size=(224,224))
    image=np.array(image)
    image=image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
    pred=np.argmax(model.predict(image))
    dic={0:'Diseased Cotton Leaf',1:'Diseased Cotton Plant', 2:'Fresh Cotton Leaf', 3:'Fresh Cotton Plant'}
    for key, val in dic.items():
        if pred==key:
            x=val
    image_path=os.path.join(app.config['UPLOAD_FOLDER'],str(image_path).replace('static/images/',''))
    return render_template('index.html', prediction='{}'.format(x), image_path='{}'.format((image_path)))
if __name__=='__main__':
    app.run(debug=True)