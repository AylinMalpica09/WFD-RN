from keras.models import load_model
import cv2
import numpy as np

model = load_model('modelo.h5')

image_path = input("por favor, ingrese la ruta de la imagen: ")

image_path = image_path.strip('"')

img =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64,64))
img = np.array(img).reshape(-1, 64, 64, 1)

#HACER LA PREDICCION 
prediction = model.predict(img)

#OBTENER LA CLASE PREDICHA
classes =['fresa','mora','guayaba','ciruela','calabaza','tomate_v']
predicted_classes = classes[np.argmax(prediction)]

print(f'La clase predicha es: {predicted_classes}')