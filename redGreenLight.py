#Import des packages
import numpy as np
import mediapipe as mp
import cv2
from playsound import playsound
import time
inFramecheck = False
inFrame = 0
cPos = 0
isInit = False
duration = 0
startT = 0
endT = 0
playerSum = 0
isCInit = False
cStartT = 0
cEndT = 0
winner = 0
isAlive = 1
thresh = 180
#declarer la fonction isVisible
def isVisible(landmarkList):
    if(landmarkList[28].visibility > 0.7) and (landmarkList[24].visibility > 0.7):
        return True
    return False
def calc_sum(landmarkList):
    tsum = 0
    for i in range(11, 32):
        tsum += (landmarkList[i].x * 480)
    return tsum
def calc_dist(landmarkList):
    return (landmarkList[28].y * 640 - landmarkList[24].y * 640)

#initialisation de la varible
capture = cv2.VideoCapture(0)

#Initialiser mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

im1 = cv2.imread("im1.png")
im2 = cv2.imread("im2.png")

currentWindow = im1

#Initialiser le webcam
while True:
    success, frm = capture.read()
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    #print(res.pose_landmarks)
    frm = cv2.blur(frm, (5, 5))
    drawing.draw_landmarks(frm, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if not(inFramecheck):
        try:
            if isVisible(res.pose_landmarks.landmark):
                inFrame = 1
                inFramecheck = True
            else:
                inFrame = 0
        except:
            print("Vous n'etes pas visible")
    if inFrame == 1:
        if not(isInit):
            playsound("greenLight.mp3")
            currentWindow = im1
            startT = time.time()
            endT = startT
            duration = np.random.randint(1, 5)
            isInit = True
        if (endT - startT) <= duration:
            try:
                m = calc_dist(res.pose_landmarks.landmark)
                if m < thresh:
                    cPos +=1
                print("Votre progression :", cPos)
            except:
                print("pas visible")
            endT = time.time()
        else:
            if cPos >= 100:
                print("Vous avez gagnez")
                winner = 1
            else:
                if not(isCInit):
                    isCInit = True
                    cStart = time.time()
                    cEndT = cStartT
                    currentWindow = im2
                    playsound("redLight.mp3")
                    playerSum = calc_sum(res.pose_landmarks.landmark)
                if (cEndT - cStart) <= 3:
                    temp = calc_sum(res.pose_landmarks.landmark)
                    if abs(temp - playerSum) > 150:
                        print("ELIMINER", abs(temp - playerSum) > 150)
                        isAlive = 0
                else:
                    isInit = False
                    isCInit = False
    else:
        cv2.putText(frm, "Placez vous dans le cadre", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4)
# Afficher la sortie final
    currentWindow = cv2.circle(currentWindow, ((55 + 6 * cPos), 280), 15, (0, 0, 255), -1)
    frm = np.concatenate((cv2.resize(frm, (800,400)), currentWindow), axis=0)
    cv2.imshow("Sortie", frm)
    #print(frm.shape) # permet de reconnaitre la dimension de la matrice frm
    #currentWindow = cv2.resize(currentWindow, (640, 480), interpolation=cv2.INTER_AREA)
    #frm = cv2.vconcat([frm, currentWindow])

    if cv2.waitKey(1) == ord("n") or isAlive == 0 or winner == 1:
        capture.release()
        cv2.destroyAllWindows()
        break
if isAlive == 0:
    cv2.putText(frm, "Vous etes éliminé", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    cv2.imshow("ELIMINATION", frm)
if winner == 1:
    cv2.putText(frm, "Vous avez gagné, Félicitation !", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("WINNER", frm)
# Active le webcam et arreter windows activity
cv2.waitKey()
