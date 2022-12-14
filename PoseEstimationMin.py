
import cv2
import mediapipe as mp
import time
import numpy as np
import os
from matplotlib import pyplot as plt
import time

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def extract_keypoints(results):
    poseL = []
    temp = [results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST.value]]
    for res in temp:
        if results.pose_landmarks.landmark:
            test = np.array([res.x, res.y, res.z, res.visibility])
            poseL.append(test)
        else:
            np.zeros(33 * 4)


    return np.concatenate([poseL]).flatten()

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

DATA_PATH = os.path.join('C:/Users/Crim/Desktop/Golf_Set_Frames')

# Actions that we try to detect
actions = np.array(['Good'])
# Fifteen videos worth of data
no_sequences = 10
# Videos are going to be 15 frames in length
sequence_length = 500

# Videos Number
videoNo = 1
folderNo = 0


for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# for i in range(10):
#     num = 'videos/' + str(videoNo) + '.mp4'
#     cap = cv2.VideoCapture(num)
#
#
#     try:
#         for frame_num in range(sequence_length):
#             # Read feed
#             success, img = cap.read()
#             image = cv2.resize(img, (1000, 1000))
#             imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             results = pose.process(imgRGB)
#
#             try:
#                 landmarks = results.pose_landmarks.landmark
#
#                 #(LEFT) UpperBody Keypoints
#                 leftEar = [landmarks[mpPose.PoseLandmark.LEFT_EAR.value].x,
#                             landmarks[mpPose.PoseLandmark.LEFT_EAR.value].y]
#                 # --- Also a stroke keypoint
#                 leftShoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
#                                     landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 # -- Also a lower body keypoint
#                 leftHip = [landmarks[mpPose.PoseLandmark.LEFT_HIP].x,
#                             landmarks[mpPose.PoseLandmark.LEFT_HIP].y]
#
#                 #(RIGHT) UpperBody Keypoints
#                 rightEar = [landmarks[mpPose.PoseLandmark.RIGHT_EAR.value].x,
#                                 landmarks[mpPose.PoseLandmark.RIGHT_EAR.value].y]
#                 #  -- Also a stroke keypoint
#                 rightShoulder = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                                     landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
#                 # -- Also a lower body keypoint
#                 rightHip = [landmarks[mpPose.PoseLandmark.RIGHT_HIP].x,
#                                 landmarks[mpPose.PoseLandmark.RIGHT_HIP].y]
#
#                 # (LEFT) Stroke Keypoints
#                 leftElbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,
#                                 landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
#                 leftWrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,
#                                 landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]
#
#                 # (RIGHT) Stroke Keypoints
#                 rightElbow = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,
#                                 landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
#                 rightWrist = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,
#                                 landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]
#
#                 #(LEFT) Lowerbody Keypoints
#                 leftKnee = [landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].x,
#                                 landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].y]
#                 leftAnkle = [landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].x,
#                              landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].y]
#                 #(RIGHT) Lowerbody Keypoints
#                 rightKnee = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,
#                                 landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
#                 rightAnkle = [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x,
#                                 landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]
#
#
#                 # () Calculate angle
#                 #UPPER BODY
#                 leftUpperBody = int(calculate_angle(leftEar, leftShoulder, leftHip))
#                 rightUpperBody = int(calculate_angle(rightEar, rightShoulder, rightHip))
#
#                 #STROKE
#                 leftStroke = int(calculate_angle(leftShoulder, leftElbow, leftWrist))
#                 rightStroke = int(calculate_angle(rightShoulder, rightElbow, rightWrist))
#
#                 #LOWER BODY
#                 leftLowerBody = int(calculate_angle(leftHip, leftKnee, leftAnkle))
#                 rightLowerBody = int(calculate_angle(rightHip, rightKnee, rightAnkle))
#
#                 #(TEST) LOWER BENDING
#                 leftLowerBendingBody = int(calculate_angle(leftShoulder, leftHip, leftKnee))
#                 rightLowerBendingBody = int(calculate_angle(rightShoulder, rightHip, rightKnee))
#             except:
#                 pass
#             if results.pose_landmarks:
#                 mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#             else:
#                 break
#
#             if success:
#                 currentFrame = image
#                 if frame_num == 0:
#                     cv2.putText(image, 'Collecting frames for Video Number: {} Frame Number {}'.format(folderNo, frame_num), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
#                     # Show to screen
#                     cv2.imshow('OpenCV Feed', image)
#                 else:
#                     cv2.putText(image, 'Collecting frames for Video Number: {} Frame Number {}'.format(folderNo, frame_num), (15, 12),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
#                     # Show to screen
#                     cv2.imshow('OpenCV Feed', image)
#
#                 keypoints = extract_keypoints(results)
#                 print(keypoints)
#                 npy_path = os.path.join(DATA_PATH, "Good", str(folderNo), str(frame_num))
#                 np.save(npy_path, keypoints)
#                 numberOfFramesCollected = frame_num
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#
#     except:
#         print("Video Stopped and collected ", numberOfFramesCollected, "of Frames.")
#         pass
#     videoNo+=1
#     folderNo+=1
#     cap.release()
#     cv2.destroyAllWindows()



for i in range(10):

    num = 'videos/' + str(videoNo) + '.mp4'
    cap = cv2.VideoCapture(num)
    videoNo += 1
    previousTime = 0
    frame = 0

    success, img = cap.read()

    #FPS COUNTER
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    #WINDOW SIZE
    img2 = cv2.resize(img, (1000, 1000))
    imgRGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    while True:
        success, img = cap.read()

        # FPS COUNTER
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # WINDOW SIZE
        img2 = cv2.resize(img, (1000, 1000))
        imgRGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        #PRINTING LANDMARKS
        # print(results.pose_landmarks)
        try:
            landmarks = results.pose_landmarks.landmark


            ###### KEYPOINTS #####
            #(LEFT) UpperBody Keypoints
            leftEar = [landmarks[mpPose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mpPose.PoseLandmark.LEFT_EAR.value].y]
            # --- Also a stroke keypoint
            leftShoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
            # -- Also a lower body keypoint
            leftHip = [landmarks[mpPose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mpPose.PoseLandmark.LEFT_HIP].y]

            #(RIGHT) UpperBody Keypoints
            rightEar = [landmarks[mpPose.PoseLandmark.RIGHT_EAR.value].x,
                            landmarks[mpPose.PoseLandmark.RIGHT_EAR.value].y]
            #  -- Also a stroke keypoint
            rightShoulder = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
            # -- Also a lower body keypoint
            rightHip = [landmarks[mpPose.PoseLandmark.RIGHT_HIP].x,
                            landmarks[mpPose.PoseLandmark.RIGHT_HIP].y]

            # (LEFT) Stroke Keypoints
            leftElbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
            leftWrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

            # (RIGHT) Stroke Keypoints
            rightElbow = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
            rightWrist = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]

            #(LEFT) Lowerbody Keypoints
            leftKnee = [landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].y]
            leftAnkle = [landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].y]
            #(RIGHT) Lowerbody Keypoints
            rightKnee = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
            rightAnkle = [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]


            # () Calculate angle
            #UPPER BODY
            leftUpperBody = int(calculate_angle(leftEar, leftShoulder, leftHip))
            rightUpperBody = int(calculate_angle(rightEar, rightShoulder, rightHip))

            #STROKE
            leftStroke = int(calculate_angle(leftShoulder, leftElbow, leftWrist))
            rightStroke = int(calculate_angle(rightShoulder, rightElbow, rightWrist))

            #LOWER BODY
            leftLowerBody = int(calculate_angle(leftHip, leftKnee, leftAnkle))
            rightLowerBody = int(calculate_angle(rightHip, rightKnee, rightAnkle))

            #(TEST) LOWER BENDING
            leftLowerBendingBody = int(calculate_angle(leftShoulder, leftHip, leftKnee))
            rightLowerBendingBody = int(calculate_angle(rightShoulder, rightHip, rightKnee))

            # print(results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST.value].z)

        except:
            pass
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img2, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        else:
            break

        ### THIS IS ORDERED BY Y AXIS
        cv2.putText(img2, "Details: ", (70, 640), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 255, 0), 2)

        cv2.putText(img2, "FPS: " + str(int(fps)), (70,660), cv2.FONT_HERSHEY_PLAIN, 1,
                             (0, 255, 255), 1)

        #UPPER BODY PUTTEXT
        cv2.putText(img2, "UPPERBODY DETAILS: ", (70, 690), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 2)
        cv2.putText(img2, "Left Upperbody Angle: " + str(leftUpperBody), (70, 720), cv2.FONT_HERSHEY_PLAIN, 1,
                   (255, 255, 0), 1)
        cv2.putText(img2, "Right Upperbody Angle: " + str(rightUpperBody), (70, 740), cv2.FONT_HERSHEY_PLAIN, 1,
                   (255, 255, 0), 1)

        #STROKE PUTTEXT
        cv2.putText(img2, "STROKE DETAILS: ", (70, 770), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 2)
        cv2.putText(img2, "Left Stroke Angle: " + str(leftStroke), (70, 800), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 0), 1)
        cv2.putText(img2, "Right Stroke Angle: " + str(rightStroke), (70, 820), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 0), 1)

        #LOWER BODY PUTTEXT
        cv2.putText(img2, "LOWERBODY DETAILS: ", (70, 850), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 2)
        # cv2.putText(img2, "Left Lowerbody Angle: " + str(leftLowerBody), (70, 880), cv2.FONT_HERSHEY_PLAIN, 1,
        #             (255, 255, 0), 1)
        # cv2.putText(img2, "Right Lowerbody Angle: " + str(rightLowerBody), (70, 900), cv2.FONT_HERSHEY_PLAIN, 1,
        #             (255, 255, 0), 1)
        cv2.putText(img2, "Left Lowerbody Angle: " + str(leftLowerBendingBody), (70, 880), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 0), 1)
        cv2.putText(img2, "Right Lowerbody Angle: " + str(rightLowerBendingBody), (70, 900), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 0), 1)

        cv2.putText(img2, "Press Q to exit Video", (70, 920), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 0), 1)

        if leftUpperBody < 160 and rightUpperBody < 160:
            cv2.putText(img2, "BENDING: TRUE", (70, 1000), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 0), 2)
        else:
            cv2.putText(img2, "BENDING: FALSE", (70, 1000), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 0), 2)



        if success:
            cv2.imshow("img", img2)
            currentFrame = img2
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

while True:
    cv2.imshow("Frame", currentFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    while True:

        for action in actions:
            print(frame)

            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    success, img = cap.read()
                    image = cv2.resize(img, (1000, 1000))
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = pose.process(imgRGB)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(frame, temp-1), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, "Ankle", str(temp1), str(frame))
                    frame += 1
                    np.save(npy_path, keypoints)
                    break


    cap.release()
    cv2.destroyAllWindows()

