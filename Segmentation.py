import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2

DATA_PATH = "../MIA23_Project1_data/" #Path to folder with Patient Folders
TEST_PATH = "../MIA23_Project1_data_test/"


def read_patient_info_file(patient_number):
    file_path = DATA_PATH + f"patient{patient_number:04d}/Info_2CH.cfg"
    with open(file_path, 'r') as f:
        return f.read()
    return ""


def read_patient_mhd_file(patient_number, file_prefix, test=False, results=False):
    if not test:
        file_path = DATA_PATH + f"patient{patient_number:04d}/patient{patient_number:04d}_{file_prefix}.mhd"
    else:
        file_path = TEST_PATH + f"Test{patient_number}/test{patient_number}_{file_prefix}.mhd"

    if results:
        file_path = TEST_PATH + f"Test{patient_number}/R_{file_prefix}_sequence.mhd"

    # Read the mhd file
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)
    
    spacing = image.GetSpacing()
    
    return array, spacing[1]/spacing[0], spacing

def write_mhd_file(arr, patient_number, file_prefix, spacing):
    file_path = TEST_PATH + f"Test{patient_number}/R_{file_prefix}_sequence.mhd"

    image = sitk.GetImageFromArray(arr)
    # spacing.reverse()
    image.SetSpacing(spacing)

    sitk.WriteImage(image, file_path, useCompression=False)


def view_image(image, aspectR, title=None):
    plt.imshow(image, cmap='gray', aspect=aspectR)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

#TAKEN FROM OPENCV DOCS   https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

def otsuValue(image):
    mask = (image > 0).astype(np.uint8)
    
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([image],[0],mask,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh

def setSubplot(subplot, image, aspectRatio, title):
    subplot.imshow(image, cmap='gray', aspect=aspectRatio)
    subplot.set_title(title)
    subplot.axis('off')

def findLargeComponents(image, constrain):
    numRegions, label, stats, centroids = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S) #inefficient
    components = []
    for i in range(numRegions):
        if i == 0: #Skip background
            continue
        if (stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT] > 6000) and ((not constrain) or (centroids[i][1] < (image.shape)[0] * 0.55)):
            components.append([centroids[i][1], i])
    return sorted(components), label #Pick topmost component
    
    
def findPrimaryComponent(imageFrame, max_iter):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    
    components = []
    label = None
    iterations = 0
    while len(components) == 0 and iterations < max_iter:
        size = 1 if iterations == 0 else 5*iterations
        closed = cv2.dilate(imageFrame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)), iterations=1)
        closed = cv2.erode(closed, kernel, iterations=1)
        closed = cv2.dilate(closed, kernel, iterations=1)
    
        components, label = findLargeComponents(closed, iterations < max_iter-1)
        iterations += 1
    
    assert(label is not None)
    closed[label != components[0][1]] = 0
    return closed
    
    
def segmentImage(image, aspectRatio, title, display, displayFinal):
    imageHeight = (image.shape)[0]
    imageWidth = (image.shape)[1]


    # Things to do:
    """
    Find right side of ventricle using convolved image
    Find bottom of ventricle using a horizontal line filter
    Put lines in those spots and mask outside image
    Something might be going wrong with findPrimaryComponent or findLargeComponents
    """

    # gamma correction
    threshold = otsuValue(image) - 25
    invGamma = 0.6
    table = np.array([((i / 255.0) ** invGamma) * 255 if i > threshold else ((i / 255.0) ** (1/invGamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(image, table)

    if display:
        fig, ax = plt.subplots(4, 3, figsize=(22, 20))
        setSubplot(ax[0, 0], image, aspectRatio, 'Original ' + title)
    
    ultrasoundMask = (image == 0) #Mask outside the ultrasound range

    dyKernel = np.ones((60,30), np.int8)
    dyKernel[:int(dyKernel.shape[0]/2),:] = -1
    yConvolved = cv2.filter2D(corrected, cv2.CV_32F, dyKernel)
    yConvolved[cv2.dilate(ultrasoundMask.astype(np.uint8), np.ones((50, 50), np.uint8), iterations=1) == 1] = 0 #Mask out dilated ultrasound part
    yConvolved = 255 * yConvolved / np.max(yConvolved) #Rescale to max = 255
    
    if display:
        setSubplot(ax[3, 0], yConvolved, aspectRatio, 'YConvolved ' + title)

    yConvolved[yConvolved < 100] = 0 #Binarize to 0/1
    yConvolved[yConvolved > 0] = 1
    yConvolved[int(.75*imageHeight):,:] = 0 #Remove strong edges from the bottom portion of image
    
    numRegions, label, stats, centroids = cv2.connectedComponentsWithStats(yConvolved.astype(np.uint8), 4, cv2.CV_32S)
    components = []
    for i in range(numRegions):
        if i == 0: #Skip background
            continue
        components.append([stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT], i])
    components = sorted(components)
    components.reverse() # largest to smallest

    for i in range(1, len(components)): # keep only largest component
        yConvolved[label == components[i][1]] = 0

    if display:
        setSubplot(ax[3,1], yConvolved, aspectRatio, 'Gradient Y max threshold')
    
    x3, y3 = np.nonzero(yConvolved)
    x3 += 15 # push a little down
    if (not np.isnan(np.average(x3))):
        corrected[int(np.average(x3)):, :] = 0 # below horizontal boundary set to 0
    
    septumKernel = np.ones((120, 60), np.int8)
    septumKernel[:,30:] = -1
    convolved = cv2.filter2D(corrected, cv2.CV_32F, septumKernel)
    convolved[cv2.dilate(ultrasoundMask.astype(np.uint8), np.ones((50, 50), np.uint8), iterations=1) == 1] = 0 #Mask out dilated ultrasound part
    convolved = 255 * convolved / np.max(convolved) #Rescale to max = 255
    
    if display:
        setSubplot(ax[0, 1], convolved, aspectRatio, 'Convolved ' + title)
    
    convolved[convolved < 100] = 0 #Binarize to 0/1
    convolved[convolved > 0] = 1
    convolved[:,int(imageWidth*0.55):] = 0 #Remove strong edges from the right of center
    numRegions, label, stats, centroids = cv2.connectedComponentsWithStats(convolved.astype(np.uint8), 4, cv2.CV_32S)
    components = []
    for i in range(numRegions):
        if i == 0: #Skip background
            continue
        components.append([stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT], i])
    components = sorted(components)
    components.reverse()

    for i in range(1, len(components)):
        convolved[label == components[i][1]] = 0
    
    if display:
        setSubplot(ax[0, 2], convolved, aspectRatio, 'Max Threshold ' + title)


    # convolve image to find right side of ventricle
    rightKernel = np.ones((120,30), np.int8)
    rightKernel[:,:int(rightKernel.shape[1]/2)] = -1
    
    convRight = cv2.filter2D(corrected, cv2.CV_32F, rightKernel)
    convRight[cv2.dilate(ultrasoundMask.astype(np.uint8), np.ones((50, 50), np.uint8), iterations=1) == 1] = 0 
    convRight = 255 * convRight / np.max(convRight) #Resacle to max = 255

    convRight[convRight < 100] = 0 #Binarize to 0/1
    convRight[convRight > 0] = 1
    convRight[:,:int(imageWidth*0.45)] = 0 #Remove strong edges from the left of center
    
    numRegions2, label2, stats2, centroids2 = cv2.connectedComponentsWithStats(convRight.astype(np.uint8), 4, cv2.CV_32S)
    components2 = []
    for i in range(numRegions2):
        if i == 0: #Skip background
            continue
        # if centroids2[i][1] > components[0][1] + 25: # only append component if it's to the right of the septum edge centroid, wouldn't work for some reason
        components2.append([stats2[i, cv2.CC_STAT_WIDTH] * stats2[i, cv2.CC_STAT_HEIGHT], i])
    components2 = sorted(components2)
    components2.reverse() # largest to smallest

    for i in range(1, len(components2)): # keep only largest component
        convRight[label2 == components2[i][1]] = 0

    if display:
        setSubplot(ax[3,2], convRight, aspectRatio, 'Right max threshold')
    
    x2, y2 = np.nonzero(convRight)
    y2 += 15 # push a little to the right
    A2 = np.vstack([x2, np.ones(len(x2))]).T
    m2, c2 = np.linalg.lstsq(A2, y2, rcond=None)[0] # estimate line at right edge
    m2 += 0.0001

    
    x, y = np.nonzero(convolved) #y is actually horizontal here
    y -= 15 # Fuzzy correction to be on edge
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0] # estimate line at left edge
    m += 0.0001
    
    
    if display:
        setSubplot(ax[1, 0], corrected, aspectRatio, 'Contrasted ' + title)
    #corrected = cv2.morphologyEx(corrected, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)), iterations=1)
    
    ret, thresholded = cv2.threshold(corrected, threshold, 255, cv2.THRESH_BINARY_INV)
    thresholded[ultrasoundMask] = 0

    RGB = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
    cv2.line(RGB, (0, int(-c/m)), (imageHeight, int((imageHeight-c)/m)), (255, 0, 0), 2)
    cv2.line(RGB, (0, int(-c2/m2)), (imageHeight, int((imageHeight-c2)/m2)), (255, 0, 0),2)

    if display:
        setSubplot(ax[1, 1], RGB, aspectRatio, 'Thresholded ' + title)
    

    # Masking off sides of boundaries and below horizontal boundary
    if (not np.isnan(np.average(x3))):
        thresholded[int(np.average(x3)):,:] = 0 # below horizontal boundary set to 0
    if np.average(y) < 0.55*imageWidth: #Only block if in left half of image, the correct septum
        pts = np.array([[0,0],[0, int(-c/m)],[imageHeight, int((imageHeight-c)/m)],[0, imageHeight]], np.int32)
        cv2.fillPoly(thresholded, [pts], (0,0,0))
    if np.average(y2) > .5 * imageWidth:
        # pts = np.array([[0, int(-c2/m2)],[imageWidth, 0],[imageWidth, imageHeight],[imageHeight, int((imageHeight-c2)/m2)]], np.int32)
        # cv2.fillPoly(thresholded, [pts], (0,0,0))
        thresholded[:, int(np.average(y2)):] = 0


    if display:
        setSubplot(ax[1, 2], thresholded, aspectRatio, 'Masked ' + title)
    
    thresholded = cv2.medianBlur(thresholded, 7)
    
    closed = findPrimaryComponent(thresholded, 15)
    if display:
        setSubplot(ax[2, 0], closed, aspectRatio, 'Opened ' + title)
    
    grown = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)), iterations=1)
    
    black = np.zeros((imageHeight + 400, imageWidth + 400))
    black[200:200+imageHeight, 200:200+imageWidth] = grown
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(27, 27)) #making this somewhat bigger results in smoother curve, but also much slower, or can use RECT shape
    extended = cv2.morphologyEx(black, cv2.MORPH_CLOSE, k, iterations=12)
    
    grown = extended[200:200 + imageHeight, 200:200+imageWidth]
    if display:
        setSubplot(ax[2, 1], grown, aspectRatio, 'Grown Component ' + title)
 
    smaller = cv2.erode(grown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6, 6)), iterations=1)
    boundary = grown - smaller
    
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    color[boundary > 0] = [255, 0, 0]
    if display:
        setSubplot(ax[2, 2], color, aspectRatio, 'Final Segmentation ' + title)
    if displayFinal:
        view_image(color, aspectRatio, 'Final Segmentation')
    if display:
        plt.show()
    return boundary, grown


def getSequenceSegmentations(patientNumber, test): 
    image, aspect, spacing = read_patient_mhd_file(patientNumber, '2CH_sequence', test)
    TwoChamber = []
    for i in range(len(image)):
        b, region = segmentImage(image[i], aspect, '2CH_sequence', False, False)
        TwoChamber.append(region.astype(np.uint8))
        
    image, aspect, spacing = read_patient_mhd_file(patientNumber, '4CH_sequence', test)
    FourChamber = []
    for i in range(len(image)):
        b, region = segmentImage(image[i], aspect, '4CH_sequence', False, False)
        FourChamber.append(region.astype(np.uint8))
    
    return TwoChamber, FourChamber, aspect