import cv2
import numpy as np

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rectCountour(coutours):  #detects small contours accourding to aera

    rectCon=[]
    for i in coutours:
        area = cv2.contourArea(i)
        print("Area",area)
        if area>50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            # print("Corner Points",len(approx))
            if len(approx)==4:
                rectCon.append(i)
    rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)

    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    # print(myPoints)
    # print(add)
    myPointsNew[0] = myPoints[np.argmin(add)] #[0,0]
    myPointsNew[3] = myPoints[np.argmax(add)] #[w,h]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  #[0,h]
    # print(diff)

    return myPointsNew


def splitBoxes(img):                                 #this function splits the row of the marks 5 ques as it appears as a matrix and to split the matrx row we are using numpy
    rows = np.vsplit(img,5)
    boxes = []  # an array to store each element of the matrix
    for r in rows:
        cols = np.hsplit(r,4)
        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split",box)
    return boxes


def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = img.shape[1] // choices  # Calculate section width based on choices
    secH = img.shape[0] // questions  # Calculate section height based on questions
    circle_radius = min(secW, secH) // 2 - 20
    circle_radius1 = min(secW, secH) // 2 - 40
    for x in range(questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2

        if grading[x] == 1:
            myColor = (0, 255, 0) #green
        else:
            myColor = (0, 0, 255) #red
            correctAns = ans[x]
            cv2.circle(img, ((correctAns*secW)+secW//2,(x*secH)+secH//2), circle_radius1, (0, 255, 0), cv2.FILLED)



        # Reduce the size of the colored circle slightly by subtracting 10 from the radius


        cv2.circle(img, (cX, cY), circle_radius, myColor, cv2.FILLED)

    return img
