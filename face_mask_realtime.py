import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing import image
# from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model('model.h5')
figure_size= 3
while True:
    _, img = cap.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in face:
        face_img = img[y:y + h, x:x + w]
        cv2.imwrite('sample.jpg', face_img)
        test_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        test_image = cv2.medianBlur(test_image, figure_size)
        test_image = image.load_img('sample.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        pred = (model.predict(test_image)[0][0] > 0.5).astype("int32")

        if pred == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, 'WARNING', ((x + w) // 2, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    cv2.imshow('Test_nhan_dien_deo_khau_trang', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
