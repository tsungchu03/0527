import cv2

import os

input = "face"

output = "face_output"

if not os.path.exists(output):

    os.makedirs(output)

face_cascade = cv2.CascadeClassifier(

    "haarcascade_frontalface_default.xml"

)

for filename in os.listdir(input):

    img_path = os.path.join(input, filename)

    img = cv2.imread(img_path)

    if img is None:

        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(img, "face", (x, y - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8, (0, 255, 0), 2)

    save_path = os.path.join(output, "detected_" + filename)

    cv2.imwrite(save_path, img)

print("處理完成")

