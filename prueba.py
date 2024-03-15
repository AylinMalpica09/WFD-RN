from flask import Flask, request, render_template
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__, static_url_path='/src', static_folder='src')

model = load_model('modelo.h5')
classes =['fresa','mora','guayaba','ciruela','calabaza','tomate_v']

def process_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = np.array(img).reshape(-1, 64, 64, 1)
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            predicted_class = process_image(file)
            return render_template('result.html', predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
