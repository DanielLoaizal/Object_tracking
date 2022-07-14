import cv2
import numpy as np
from gui_buttons import Buttons

# Initializate
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("keyboard", 20, 80)
button.add_button("cell phone", 20, 140)
button.add_button("car", 20, 200)

# Opencv CNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model\yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Global variables definition
button_person = False

# Load class lists
classes = []
with open("dnn_model\classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print(classes)

# initializate the camera
cap = cv2.VideoCapture(0)

# HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# FULL HD: 1920 X 1080

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)
            



# Create Window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    # Get frames
    ret, frame = cap.read()

    # get active button list{
    active_buttons = button.active_buttons_list()

    # object detection
    (class_ids, scores, bboxes) = model.detect(frame)

    for class_ids, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox

        if class_name in active_buttons:
            cv2.putText(img=frame, text=classes[class_ids], org=(x,y-5), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(200,0,50), thickness=2)
            cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w, y+h), color=(200,0,50), thickness=3)
            center_x = int((2*x+w)/2)
            center_y = int((2*y+h)/2)
            # print(center_x, center_y)
            cv2.circle(img=frame, center=(center_x,center_y), radius=1, color=(0, 0, 255), thickness=-1)

    # Display Buttons

    button.display_buttons(frame)

    # # Create Button
    # # cv2.rectangle(img=frame, pt1=(20,20), pt2=(220,70), color=(0,0,200), thickness=-1)
    # polygon = np.array([[(20,20), (220, 20), (220, 70), (20, 70)]])
    # cv2.fillPoly(img=frame, pts=polygon, color=(0,0,200))
    # cv2.putText(img=frame, text=("person"), org=(30,60), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255,255,255), thickness=3)

    # print(bbox)

    cv2.imshow("Frame", frame)
    cv2.waitKey(100)