import cv2
from codigos.tracker import *

# create tracker object
tracker = EuclideanDistTracker()


cap = cv2.VideoCapture("Object_Tracking\highway.mp4")

# Object detection from a stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)



while True:

    ret, frame = cap.read()
    # Extraction of the region of interest
    roi = frame[340:680,500:800]


    # Step 1. aaaaaaObject detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(src=mask, thresh=254, maxval=255, type=cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in  contours:

        # Calculate Area and remove small elements:
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(image=roi, contours=[cnt], contourIdx=-1, color=(0,255,0), thickness=2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img=roi, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=1)
            detections.append([x,y,w,h])
    
    # Step 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.putText(img=roi, text=str(id), org=(x,y-15), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255,0,0), thickness=2)
        cv2.rectangle(img=roi, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=3)




    # Imagen show
    cv2.imshow("ROI", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

