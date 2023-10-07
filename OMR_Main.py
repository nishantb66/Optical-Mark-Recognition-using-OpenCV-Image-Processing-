import cv2
import numpy as np
import utlis

##################################
path = "1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 4
ans = [0,2,0,0,3]
# ans1 = [2,2,1,0,1]
##################################


img = cv2.imread(path)
img = cv2.resize(img,(widthImg, heightImg)) #resizing the image
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()


#PREPROCESSING OF THE IMAGE
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5),1)
#detect edges
imgCanny = cv2.Canny(imgBlur,10,50)
#find countours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),2)
#find rectangles
rectCon=utlis.rectCountour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
gradePoints = utlis.getCornerPoints(rectCon[1])

# print(biggestContour)
if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour,-1,(0,255,0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255,0,0), 20)

    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    #bird eye view of largest countour
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    #birds eye view of grade point box
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
    imgGradeDisplay= cv2.warpPerspective(img,matrixG,(325,150))
    # cv2.imshow("Grade",imgGradeDisplay)

    #Apply Thresholding
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray,100,255,cv2.THRESH_BINARY_INV)[1]


    boxes = utlis.splitBoxes(imgThresh)
    # cv2.imshow("Test",boxes[2])
    # print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))


    #Getting non zero pixels values of each box
    myPixelVal = np.zeros((questions, choices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices):
            countR += 1
            countC = 0
    # print(myPixelVal)


    #finding index values of the markings
    myIndex = []
    for x in range (0,questions):
        arr = myPixelVal[x]
        # print("arr",arr)
        myIndeVal = np.where(arr==np.amax(arr)) #finding the maximum pixel vale box in the array of 4 choices
        # print(myIndeVal[0])
        myIndex.append(myIndeVal[0][0])
    # print(myIndex)


    # Grading
    grading = []
    for x in range (0,questions):
        if ans[x] == myIndex[x]:
            grading.append(1)  #if correct i.e matched then put 1
        else:
            grading.append(0)  #if wrong i.e not matched then put 0
    # print(grading)
    score = (sum(grading)/questions)*100  #final grade
    print(score)


    #Displaying Answers = the main function is in the utlis file
    imgResult = imgWarpColored.copy()
    imgResult = utlis.showAnswers(imgResult,myIndex,grading,ans,questions,choices)
    # mgResult = utlis.showAnswers(imgResult,myIndex,grading,ans1,questions,choices)



    #these are the correction green and red points that we will place in our original image of omr
    imRawDrawings = np.zeros_like(imgWarpColored)
    imRawDrawings = utlis.showAnswers(imRawDrawings,myIndex,grading,ans,questions,choices)
    #the below code will chnage the placement style of the colored points such that we can place them on our original image.
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imRawDrawings, invMatrix, (widthImg, heightImg))


    #To display the grade
    imgRawGrade=  np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score))+"%",(50,100),cv2.FONT_HERSHEY_COMPLEX,4,(0,255,255),4)
    invMatrixG = cv2.getPerspectiveTransform(ptG2,ptG1)
    imgInvGradeDisplay= cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))



    #combining the imgInvWarp on original image
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)





imgBlank = np.zeros_like(img)
imageArray = ([img,imgGray,imgBlur,imgCanny],
              [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
              [imgResult,imRawDrawings,imgInvWarp,imgFinal])

lables = [["Original","Gray","Blur","Canny"],
          ["Contours","Biggest Con","Warp","Threshold"],
          ["Results","Raw Drawing","Inverse Warp","Final"]]
imgStacked = utlis.stackImages(imageArray,0.3,lables)

cv2.imshow("Final Results",imgFinal)


cv2.imshow("Stacked Images", imgStacked)
cv2.waitKey(0)