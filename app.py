import cv2
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import pyautogui

from csv import writer

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

#mediapipe initializer
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

#global variable
scroll_up_x = []
scroll_up_y = []
scroll_down_x = []
scroll_down_y = []
swipe_right_x = []
swipe_right_y = []
swipe_left_x = []
swipe_left_y = []

def clear_lists(list1, list2):
    list1.clear()
    list2.clear()

def clear_all_list_except(exception):
    if exception != 0:
        scroll_up_x.clear()
        scroll_up_y.clear()
    
    if exception != 1:
        scroll_down_x.clear()
        scroll_down_y.clear()

    if exception != 2:
        swipe_right_x.clear()
        swipe_right_y.clear()

    if exception != 3:
        swipe_left_x.clear()
        swipe_left_y.clear()

def mode_now(key):
    if key == 100: #d
        return 1 #debug
    elif key == 97:
        return 0

def draw_text(id_pose, frame, x_minPix, y_minPix, predict):
    #draw to the image
    if id_pose==0:
        frame = cv2.putText(frame,'pointer',(x_minPix,y_minPix-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,bottomLeftOrigin=False)
    elif id_pose==1:
        frame = cv2.putText(frame,'scroll',(x_minPix,y_minPix-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,bottomLeftOrigin=False)
    elif id_pose==2:
        frame = cv2.putText(frame,'slide',(x_minPix,y_minPix-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,bottomLeftOrigin=False)
    elif id_pose==3:
        frame = cv2.putText(frame,'capture',(x_minPix,y_minPix-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,bottomLeftOrigin=False)
    elif id_pose==4:
        frame = cv2.putText(frame,'none',(x_minPix,y_minPix-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,bottomLeftOrigin=False)
    elif id_pose==5:
        frame = cv2.putText(frame,'ok',(x_minPix,y_minPix-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,bottomLeftOrigin=False)

    frame = cv2.putText(frame,np.array2string(predict),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,bottomLeftOrigin=False)

def predict_pose(handlandmark, data):
    #labeling image
    model_dir = '/Users/feliciarulita/Documents/practice/project5_HandGesture/model/poseClassifier/keypointsCoordinate.keras'
    load_model = tf.keras.models.load_model(model_dir)

    predict = load_model.predict(data)

    print(predict)

    #print result
    max_value = np.max(predict[0])
    if max_value > 0.4:
        id_pose = np.where(predict[0]==max_value)[0][0]
    else:
        id_pose = 10

    print("id pose {}".format(id_pose))

    return id_pose, predict

def capture_data(key, x, y, path):
    #merge key, x ,and y into a list
    points = []
    points.append(key)
    points = points + x + y
    with open(path,'a') as file:
        writerObject = writer(file)

        writerObject.writerow(points)

        file.close()


def normalize_tip_coordinate_path(list1, list2):
    standard_x = min(list1)
    standard_y = min(list2)
    


def normalize_coordinate(hand_landmarks):
    x = []
    y = []
    standard_x = hand_landmarks.landmark[0].x
    standard_y = hand_landmarks.landmark[1].y #change to 0 HAVE TO CHANGE THE CSV AGAIN OMG
    print("wrist coord = {}".format(hand_landmarks.landmark[0].x))
    for lm in enumerate(hand_landmarks.landmark):
        print(lm[1])
        normalized_x = (lm[1].x - standard_x)
        normalized_y = (lm[1].y - standard_y)

        x.append(normalized_x)
        y.append(normalized_y)

    print("Coordinate x = {}".format(x))
    print("Coordinate y = {}".format(y))

    all_data = x + y

    #convert to np array
    arr = np.array(all_data).reshape(1,42)

    return x,y,arr

def bounding_box(hand_landmarks,height,width):
    #bounding box
    min_x = min([lm[1].x for lm in enumerate(hand_landmarks.landmark)])
    min_y = min([lm[1].y for lm in enumerate(hand_landmarks.landmark)])
    max_x = max([lm[1].x for lm in enumerate(hand_landmarks.landmark)])
    max_y = max([lm[1].y for lm in enumerate(hand_landmarks.landmark)])

    for lm in enumerate(hand_landmarks.landmark):
        print(lm[1])

    #convert to pixel
    x_minPix = int(min_x * width)
    y_minPix = int(min_y * height)
    x_maxPix = int(max_x * width)
    y_maxPix = int(max_y * height)

    return x_minPix,y_minPix,x_maxPix,y_maxPix,min_x,min_y

def print_result(result: HandLandmarkerResult, outputImage: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    print(' ')

def main():
    #debug
    check = False

    #configuration
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./model/hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result
    )
    hands = mp_hands.Hands(
        static_image_mode= False,
        max_num_hands= 2,
        min_detection_confidence= 0.5
    )
    mode = 0

    #open default camera
    try:
        cam = cv2.VideoCapture(1)
        print("Camera is opened succesfully")
    except:
        print("Camera is not opened")

    with HandLandmarker.create_from_options(options) as landmarker:
        while(True):
            ret, frame = cam.read()
            cam.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS


            #flip the camera
            #frame = cv2.flip(frame,1)

            #convert to mediapipe's image object
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #have all the coordinates
            detected_hands = hands.process(frame)
            #convert back to opencv's
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Detect asynchronously, using the timestamp of the frame
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            #wait for key press in 1 milisecond
            key = cv2.waitKey(1)
            #d for training, a for testing
            if key == 100:
                mode = 1
            elif key == 97:
                mode = 0

            if detected_hands.multi_hand_landmarks:
                for hand_landmarks in detected_hands.multi_hand_landmarks:
                    
                    #passing parameters for drawing the landmarks
                    drawing.draw_landmarks(
                        frame, #return the override frame to the frame itself
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style()
                    )

                    # Draw the bounding box
                    height, width, _ = frame.shape
                    x_minPix, y_minPix, x_maxPix, y_maxPix, min_x, min_y = bounding_box(hand_landmarks, height, width)
                    cv2.rectangle(frame, (x_minPix, y_minPix), (x_maxPix, y_maxPix), (0, 255, 0), 2)

                    print("x_min = {}".format(min_x))
                    print("y_min = {}".format(min_y))

                    #Normalize Coordinate
                    x, y, arr_data = normalize_coordinate(hand_landmarks)
                    path = '/Users/feliciarulita/Documents/practice/project5_HandGesture/model/poseClassifier/keypointsCoordinate.csv'

                    #capturing data
                    if key==48: #0 for taking data POINT
                        print("Take data 0")
                        capture_data(0, x, y, path)
                    elif key==49: #1 for taking data SCROLL
                        print("Take data 1")
                        capture_data(1, x, y, path)
                    elif key==50: #2 for taking data SWIPE
                        print("Take data 2")
                        capture_data(2, x, y, path)
                    elif key==51: #3 for taking data SCREENSHOT
                        print("Take data 3")
                        capture_data(3, x, y, path)
                    elif key==52: #4 for taking data NONE
                        print("Take data 4")
                        capture_data(4, x, y, path)
                    elif key==53: #5 for taking data OK
                        print("Take data 5")
                        capture_data(5, x, y, path)
                    elif key==54: #6 for taking data FOUR
                        print("Take data 6")
                        capture_data(6, x, y, path)


                    # print(arr_data.shape)
                    
                    #using the trained model to read the pose
                    id_pose, predict = predict_pose(hand_landmarks, arr_data)

                    #draw text to frame
                    draw_text(id_pose, frame, x_minPix, y_minPix, predict)

                    #operating GUI 
                    path = '/Users/feliciarulita/Documents/practice/project5_HandGesture/model/poseClassifier/tipPathCoordinate.csv'
                    print("mode = {}".format(mode))
                    print("id pose = {}".format(id_pose))
                    if mode == 1:
                        if id_pose == 1: #for scroll up
                            #put the tip's coordinate into a list
                            scroll_up_x.append(hand_landmarks.landmark[8].x)
                            scroll_up_y.append(hand_landmarks.landmark[8].y)

                            #if the list's length is 15 then put to csv 
                            if len(scroll_up_x) == 15:
                                # capture_data(key, scroll_up_x, scroll_up_y, path)
                                #clear these list
                                clear_lists(scroll_up_x, scroll_up_y)

                            #always empty other pose's list
                            clear_all_list_except(0)
                        print("result scroll")
                        print(scroll_up_x)
                        print(scroll_down_x)
                        print(swipe_right_x)
                        print(swipe_left_y)

                    # check = True 

            #show the result
            cv2.imshow('video',frame)
            
            if key==27 or check==True: #ESC
                print("Quit Camera")
                break

    cam.release()
    cv2.destroyAllWindows()

main()