import numpy as np
import matplotlib.pyplot as plt
import cv2

def getDirection(start, end):
    return [end[0] - start[0], end[1] - start[1]]

def cosAngle(v1, v2):
    return (v1[0]*v2[0] + v1[1]*v2[1]) / (np.sqrt(v1[0]*v1[0] + v1[1]*v1[1]) * np.sqrt(v2[0]*v2[0] + v2[1]*v2[1]))

def findNextPoint(contour, currentPoint, previousDirection):
    candidates = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if 0 <= currentPoint[0] + i < contour.shape[0] and 0 <= currentPoint[1] + j < contour.shape[1] and contour[currentPoint[0] + i, currentPoint[1] + j] > 0:
                candidates.append([currentPoint[0] + i, currentPoint[1] + j])
    
    bestIndex = -1
    bestValue = -2
    for i in range(len(candidates)):
        angle = cosAngle(previousDirection, getDirection(currentPoint, candidates[i]))
        if angle > bestValue:
            bestValue = angle
            bestIndex = i
        
    return candidates[bestIndex]

def getContourPointsCW(contour):
    originalPointsY, originalPointsX = np.nonzero(contour)

    originalPointsY = np.maximum(0, np.minimum(contour.shape[0], originalPointsY)) #Bound to image size
    originalPointsX = np.maximum(0, np.minimum(contour.shape[1], originalPointsX))
    
    index = np.argmin(originalPointsY)
    startingPoint = [originalPointsY[index], originalPointsX[index]] #Topmost point
    
    pointList = [startingPoint, findNextPoint(contour, startingPoint, [0, 1])]
    while True:
        direction = getDirection(pointList[-2], pointList[-1])
        newPoint = findNextPoint(contour, pointList[-1], direction)
        if newPoint == pointList[0]:
            break
        else:
            pointList.append(newPoint)
    return pointList
    

def resampleContour(contour, N):
    originalPoints = getContourPointsCW(contour)
    ys = [x[0] for x in originalPoints]
    xs = [x[1] for x in originalPoints]
    
    originalT = np.arange(len(originalPoints))
    
    newT = np.linspace(0, len(originalPoints), N, False)
    xInterp = np.interp(newT, originalT, xs).astype(int)
    yInterp = np.interp(newT, originalT, ys).astype(int)

    #return np.stack((xInterp, yInterp), axis=1).reshape(-1, 2)
    return yInterp, xInterp
    
    
def resampleAllContours(sequence):
    newSequence = []
    for frame in sequence:
        edges = cv2.Canny(frame, 50, 200).astype(np.uint8)
        Yinterp, Xinterp = resampleContour(edges, 200)
        newSequence.append(np.array([Yinterp, Xinterp]))
    return newSequence
    
    
def interpolateBetweenContours(cStart, cEnd, t): #0 <= t <= 1
    return (1-t)*cStart + t*cEnd


def interpolateSequence(sequence, numFrames):
    control = np.linspace(0, len(sequence) - 1, numFrames, True)
    fullSequence = []
    for i in control:
        fullSequence.append(interpolateBetweenContours(sequence[int(np.floor(i))], sequence[int(np.ceil(i))], i - np.floor(i)))
    return fullSequence



def getLeftAndRightContours(c):
    Y = c[0]
    X = c[1]
    index = np.argmin(Y)
    startingPoint = [Y[index], X[index]] #Topmost point
    
    Left = [[], []]
    Right = [[], []]
    for i in range(len(X)):
        if X[i] < startingPoint[1]:
            Left[0].append(Y[i])
            Left[1].append(X[i])
        else:
            Right[0].append(Y[i])
            Right[1].append(X[i])
    Left[0].append(Right[0][0])
    Left[1].append(Right[1][0])
    
    #twoLeft[0] = np.insert(twoLeft[0], 0, twoRight[0][-1])
    #twoLeft[1] = np.insert(twoLeft[1], 0, twoRight[1][-1])
    
    return np.array(Left), np.array(Right), startingPoint
    
def alignContours(two, four, drawPlot):
    twoLeft, twoRight, twoTop = getLeftAndRightContours(two)
    fourLeft, fourRight, fourTop = getLeftAndRightContours(four)
    
    transform = getDirection(fourTop, twoTop)
    
    fourLeft[1] += transform[1]
    fourRight[1] += transform[1]
    fourLeft[0] += transform[0]
    fourRight[0] += transform[0]
    
    if drawPlot:
        plt.plot(twoLeft[1], -twoLeft[0], color='red')
        plt.plot(twoRight[1], -twoRight[0], color='darkred')

        plt.plot(fourLeft[1], -fourLeft[0], color='blue')
        plt.plot(fourRight[1], -fourRight[0],  color='lightblue')
    return twoLeft, twoRight, fourLeft, fourRight, twoTop
    
    
def resamplePath(contour, N):
    ys = contour[0]
    xs = contour[1]
    
    originalT = np.arange(contour[0].size)
    
    newT = np.linspace(0, contour[0].size, N, False)
    xInterp = np.interp(newT, originalT, xs).astype(int)
    yInterp = np.interp(newT, originalT, ys).astype(int)

    return np.array([yInterp, xInterp])
    
def generatePointCloud(two, four, intermediateAngles, aspectRatio, drawPlot):
    twoLeft, twoRight, fourLeft, fourRight, twoTop = alignContours(two, four, drawPlot)
    rotationCenter = twoTop[1]
    resampleNumber = 50
    twoLeft = resamplePath(twoLeft, resampleNumber)
    fourLeft = resamplePath(fourLeft, resampleNumber)
    twoRight = resamplePath(twoRight, resampleNumber)
    fourRight = resamplePath(fourRight, resampleNumber)
    points = []
    for i in range(-1, intermediateAngles):
        angle = 90 * (i + 1) / (intermediateAngles + 1)
        interpWeight = angle / 90
        
        interp = interpolateBetweenContours(twoLeft, fourLeft, interpWeight)
        #rotate around y axis, y is constant
        for j in range(len(interp[0])):
            radius = abs(interp[1][j] - rotationCenter)
            points.append([(-radius * np.cos(angle * np.pi / 180)) + rotationCenter, interp[0][j] * aspectRatio, -radius * np.sin(angle * np.pi / 180)]) #Now in x, y, z form
    
    for i in range(-1, intermediateAngles):
        angle = 90 * (i + 1) / (intermediateAngles + 1)
        interpWeight = angle / 90
        
        flippedLeft = np.array([fourLeft[0], 2*rotationCenter - fourLeft[1]])
        interp = interpolateBetweenContours(flippedLeft, np.flip(twoRight, axis = 1), interpWeight)
        #rotate around y axis, y is constant
        for j in range(len(interp[0])):
            radius = abs(interp[1][j] - rotationCenter)
            points.append([(radius * np.sin(angle * np.pi / 180)) + rotationCenter, interp[0][j] * aspectRatio, -radius * np.cos(angle * np.pi / 180)]) #Now in x, y, z form
    
    for i in range(-1, intermediateAngles):
        angle = 90 * (i + 1) / (intermediateAngles + 1)
        interpWeight = angle / 90
        
        interp = interpolateBetweenContours(twoRight, fourRight, interpWeight)
        #rotate around y axis, y is constant
        for j in range(len(interp[0])):
            radius = abs(interp[1][j] - rotationCenter)
            points.append([(radius * np.cos(angle * np.pi / 180)) + rotationCenter, interp[0][j] * aspectRatio, radius * np.sin(angle * np.pi / 180)]) #Now in x, y, z form
    
    
    for i in range(-1, intermediateAngles):
        angle = 90 * (i + 1) / (intermediateAngles + 1)
        interpWeight = angle / 90
        
        flippedRight = np.array([fourRight[0], 2*rotationCenter - fourRight[1]])
        interp = interpolateBetweenContours(flippedRight, np.flip(twoLeft, axis = 1), interpWeight)
        #rotate around y axis, y is constant
        for j in range(len(interp[0])):
            radius = abs(interp[1][j] - rotationCenter)
            points.append([(-radius * np.sin(angle * np.pi / 180)) + rotationCenter, interp[0][j] * aspectRatio, radius * np.cos(angle * np.pi / 180)]) #Now in x, y, z form
    
    return points