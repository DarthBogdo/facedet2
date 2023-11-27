import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # defining the capture object | CAP_DSHOW is the video source

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # setting parameters
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0  # we have to check for matches once in a while to not overload the program

face_match = False

reference_img = cv2.imread("WIN_20231121_01_09_36_Pro.jpg") # load the reference image

def face_check(frame): #checks if the reference image and the current frame have the same name on them
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False
while True:
    ret, frame = cap.read() # we use a return value to determine if it has returned anything

    if ret:
        if counter % 30 ==0:
            try:
                threading.Thread(target=face_check, args = (frame.copy(),)).start()
            except ValueError:
                pass

        counter += 1

        if face_match:
            cv2.putText(frame, 'MATCH', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        else:
            cv2.putText(frame, 'NO MATCH', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('video', frame)

    key = cv2.waitKey(1)  # process user input
    if key == ord('q'):
        break
cv2.destroyAllWindows()


