import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import vlc
import train as train
import sys, webbrowser, datetime

def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
                                shape[33],    # Nose tip
                                shape[8],     # Chin
                                shape[45],    # Left eye left corner
                                shape[36],    # Right eye right corne
                                shape[54],    # Left Mouth corner
                                shape[48]     # Right mouth corner
                            ], dtype="double")
    
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])
    
    # Camera internals
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])

def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))

#EAR -> Eye Aspect ratio
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))

def writeEyes(leftEye, rightEye, img):
    if img is None:
        print("Error: Frame is empty!")
        return

    try:
        # Left Eye
        (x, y, w, h) = cv2.boundingRect(np.array([leftEye]))
        if w == 0 or h == 0:
            print("Error: Left eye region is empty. Skipping saving.")
        else:
            leftEyeImg = img[y:y+h, x:x+w]
            cv2.imwrite('left-eye.jpg', leftEyeImg)

        # Right Eye
        (x, y, w, h) = cv2.boundingRect(np.array([rightEye]))
        if w == 0 or h == 0:
            print("Error: Right eye region is empty. Skipping saving.")
        else:
            rightEyeImg = img[y:y+h, x:x+w]
            cv2.imwrite('right-eye.jpg', rightEyeImg)

    except Exception as e:
        print(f"Error in writeEyes function: {e}")

# open_avg = train.getAvg()
# close_avg = train.getAvg()

alert = vlc.MediaPlayer(r"C:\\Users\\appua\\OneDrive\Desktop\driverdrovness\driver-drowsiness-detection-master\driver-drowsiness-detection-master\\alert-sound.mp3")

frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5

close_thresh = 0.3#(close_avg+open_avg)/2.0
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1

# print(close_thresh)

capture = cv2.VideoCapture(0)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\\Users\\appua\\OneDrive\Desktop\driverdrovness\driver-drowsiness-detection-master\driver-drowsiness-detection-master\shape_predictor_68_face_landmarks.dat")  # Corrected file name
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while(True):
    ret, frame = capture.read()
    size = frame.shape
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    rects = detector(gray, 0)
    if len(rects) == 0:
        print("No face detected! Skipping eye processing.")
        continue 
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # print("Mouth Open Ratio", yawn(shape[mStart:mEnd]))
        leftEAR = ear(leftEye) #Get the left eye aspect ratio
        rightEAR = ear(rightEye) #Get the right eye aspect ratio
        avgEAR = (leftEAR+rightEAR)/2.0
        eyeContourColor = (255, 255, 255)

        if(yawn(shape[mStart:mEnd])>0.6):
            cv2.putText(gray, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            yawn_countdown=1

        if(avgEAR<close_thresh):
            flag+=1
            eyeContourColor = (0,255,255)
            print(flag)
            if(yawn_countdown and flag>=frame_thresh_3):
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Drowsy after yawn", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_2 and getFaceDirection(shape, size)<0):
                eyeContourColor = (255, 0, 0)
                cv2.putText(gray, "Drowsy (Body Posture)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_1):
                eyeContourColor = (0, 0, 255)
                cv2.putText(gray, "Drowsy (Normal)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
        elif(avgEAR>close_thresh and flag):
            print("Flag reseted to 0")
            alert.stop()
            yawn_countdown=0
            map_flag=1
            flag=0

        if(map_counter>=3):
            map_flag=1
            map_counter=0
            vlc.MediaPlayer(r"C:\\Users\\appua\\OneDrive\Desktop\driverdrovness\driver-drowsiness-detection-master\driver-drowsiness-detection-master\\focus.mp3").play()
            webbrowser.open("https://www.google.com/maps/search/hotels+or+hotels+near+me")

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)
    if(avgEAR>close_thresh):
        alert.stop()
    cv2.imshow('Driver', gray)
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()
