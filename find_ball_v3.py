import cv2, time
import numpy as np
from kalmanfilter import KalmanFilter
from reward_functions import reward
from track import Track
from circle import Circle


video = cv2.VideoCapture('data\\point.mp4')
scaledown = .7  #Valeur de réduction de taille
precedant = None
global N         #Number of points in memory
N = 5
global tracklist
tracklist = []  #Liste des trajectoires possibles par score décroissant

#Distance between 2 points (xa, ya) and (xb, yb)
def dist(pa, pb):
    return np.sqrt((pb[0] - pa[0])**2 + (pb[1] - pa[1])**2)

#Angle formed by the line (pa pb)
def angle(pa, pb):
    return np.angle(complex(pb[0] - pa[0], pb[1] - pa[1]))



''' 
Score criteria:
    Big speed
    Top of screen
    Corresponding angle / position (KALMAN FILTER)
    Corresponding radius
    Color ?
    IS CIRCLE
        Height / Width ratio
        Compactness: Perimeter / Area
        Hough Detection
        Image comparison
'''


#Give a score to a contour regarding only its shape and position
'''
What is looked at:
    Height ratio (height comared to other contours)
        smaller = better
    Contour area ??
        usually between 300 and 800 (depends a lot on speed)
        minimum seen : 40
        maximim seen : 1200
    Height / Width ratio
        usually between 0.7 and 1.8
        minimum seen : 0.5
        maximum seen : 3.2
    Extent : Object area / Bounding rectangle area
        ususlly between 0.5 and 0.75
        minimum seen : 0.25
        maximum seen : 0.85
    Compactness: Perimeter / Area
        usually between 0.15 and 0.35
        minimum seen : 0.13
        maximum seen : 0.58
    Solidity : Contour Area / Convex Hull Area
        usually between 0.8 and 1.0
        minimum seen : 0.6
        maximum seen : 1.0

    Hough Detection
    Image comparison
    COCO image detection
'''
def contourScore(contour, miny, maxy, meany):  #https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c
    area = cv2.contourArea(contour)
    permieter = cv2.arcLength(contour, True)
    x,y,width,height = cv2.boundingRect(contour)
    center,radius = cv2.minEnclosingCircle(contour)
    hull = cv2.convexHull(contour)       #Convex shape that contains everything
    hull_area = cv2.contourArea(hull)

    aspect_ratio = height / width
    score1 = reward(aspect_ratio, 0.5, 0.7, 1.8, 3.2)
    extent = area / (width*height)
    score2 = reward(extent, 0.25, 0.5, 0.75, 0.85)
    if area:
        compactness = permieter / area
    else:
        compactness = 0
    score3 = reward(compactness, 0.13, 0.15, 0.35, 0.58)
    if hull_area:
        solidity = area / hull_area
    else:
        solidity = 0
    score4 = reward(solidity, 0.6, 0.8, 1, 2)
    heightratio = (center[1] - miny) / (meany - miny + 1)  #"+1" to avoid 0 division
    score5 = -heightratio

    totalscore = score1 + score2 + score3 + score4 + score5

    return totalscore


#Checks if an image contains a principal circular object
def findcircles(img, contour, x, y, height, width):
    cpy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    edges = cv2.Canny(img, 100, 200)
    circles = []
    
    # Hough circles method (bad)
    # hough_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=10, minRadius=0, maxRadius=200)
    # if hough_circles is not None:
    #     circles = np.uint16(np.around(hough_circles))
    #     for (x, y, r) in circles[0, :]:
    #         cv2.circle(cpy, (x, y), r, (255,0,0), 1)
    #         cv2.circle(cpy, (x, y), 0, (0,0,255), 1)

    #Find contours method
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        hull = cv2.convexHull(contour)
        center, radius = cv2.minEnclosingCircle(hull)

        hull_area = cv2.contourArea(hull)
        circle_area = np.pi*radius**2
        
        area_ratio = hull_area/circle_area
        
        if area_ratio >= 0.6:
            circles.append(((round(center[0] + x), round(center[1] + y)), round(radius), area_ratio))


        #Secondary info
        # hull_perimeter = cv2.arcLength(hull, True)
        # perimeter_ratio = hull_perimeter/hull_area

        # print(round(area_ratio, 2), end = " ")

        # if area_ratio >= 0.9:
        #     cv2.circle(cpy, tuple(map(int, center)), int(radius), (0, 255, 0), 1)
        # elif area_ratio >= 0.8:
        #     cv2.circle(cpy, tuple(map(int, center)), int(radius), (0, 200, 100), 1)
        # elif area_ratio >= 0.7:
        #     cv2.circle(cpy, tuple(map(int, center)), int(radius), (0, 100, 200), 1)
        # elif area_ratio >= 0.6:
        #     cv2.circle(cpy, tuple(map(int, center)), int(radius), (0, 50, 255), 1)

    # print("")  #Retour à la ligne
    # # cv2.drawContours(cpy, contours, -1, (255, 0, 0), 1)
    # cv2.imshow("zone", cv2.resize(cpy, [width*6, height*6]))
    # cv2.imshow("cote", cv2.resize(edges, [width*6, height*6]))
    # if cv2.waitKey(0) & 0xFF == 13:   #Skip rest of zones
    #     cv2.destroyWindow("zone")
    #     cv2.destroyWindow("cote")
    #     return circles
    # cv2.destroyWindow("zone")
    # cv2.destroyWindow("cote")

    return circles


def frame_advance(frame, points):
    global tracklist
    new = []
    for p in points:
        check = False
        for t in tracklist:
            if t.connect(p):
                # print("Connect", t, p)
                check = True
                t.check = True
                # print("Add 1")
                new.append(t.copy().add_point(p))
        if not check:
            # print("Add 2")
            new.append(track(N).add_point(p))
    for t in tracklist:
        # print("check", t.check)
        if not t.check:
            # print("Add 3")
            b = t.idle()
            if b:  #Si b a un score < 0, il est supprimé
                new.append(b)
        else:
            t.check = False

        print(t.last_point, t.predicted)
        cv2.circle(frame, tuple(map(int,t.last_point)), 15, (255, 0, 0), 4)
        cv2.circle(frame, t.predicted, 15, (0, 0, 255), 4)
    tracklist = new
    # print(len(tracklist))
    print(tracklist)





if video.isOpened():  #Get Properties
    width  = video.get(3)
    widthscale = int(width * scaledown)
    height = video.get(4)
    heightscale = int(height * scaledown)
    fps = video.get(5)
    frame_count = video.get(7)
    print(width, height, fps, frame_count)





while video.isOpened():
    ######################################################
    #### Read frame and close at the end of the video ####
    ######################################################
    check, frame = video.read()
    if not check:
        break

    ##########################################
    #### Preprocessing (grayscale + blur) ####
    ##########################################
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Get gray image
    gray = cv2.GaussianBlur(gray,(31,31), 0)  #Gaussian size needs to be odd
    if precedant is None:
        precedant = gray

    delta = cv2.absdiff(precedant, gray) #Get difference
    precedant = gray

    #############################################
    #### Get binary of everything that moves ####
    #############################################
    LOWPASS = 20
    HIGHPASS = 255
    threshold = cv2.threshold(delta, LOWPASS, HIGHPASS, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None,iterations = 10)  #Get larger whites
    threshold = cv2.erode(threshold, None, iterations = 10)
    
    #########################################################
    #### Get image of everything that moves (useless ??) ####
    #########################################################
    # result = cv2.bitwise_and(frame, frame, mask=threshold)
    # grayresult = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) #Get gray image


    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxy, miny, meany = (0, height, 0)  #y va de 0 à height

    for contour in contours:
        y = cv2.minEnclosingCircle(contour)[0][1]  #Ordonnée du centre
        if y < miny:
            miny = y
        if y > maxy:
            maxy = y
        meany += y #On fait d'abord la somme
    if contours:  #Avoid division by 0
        meany /= len(contours)

    for contour in contours:   #https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html
        #Find circles in each movement area
        x,y,width,height = cv2.boundingRect(contour)
        zone = frame[y:y+height, x:x+width]
        circles = findcircles(zone, contour, x, y, height, width)

        for c in circles:
            if c[2] >= 0.9:
                cv2.circle(frame, c[0], c[1], (0, 255, 0), 3)
            elif c[2] >= 0.8:
                cv2.circle(frame, c[0], c[1], (0, 200, 100), 3)
            elif c[2] >= 0.7:
                cv2.circle(frame, c[0], c[1], (0, 100, 200), 3)
            elif c[2] >= 0.6:
                cv2.circle(frame, c[0], c[1], (0, 50, 255), 3)


        
        #Contour score by raw shape
        # center, radius = cv2.minEnclosingCircle(contour)
        # # cv2.circle(frame, tuple(map(int, center)), int(radius), (255, 0, 0), 3)
        # score = contourScore(contour, miny, maxy, meany)
        # if score > 0 :
        #     color = (0, 255, 0)
        # else:
        #     color = (0, 0, 255)
        # cv2.putText(frame, str(round(score,2)), tuple(map(int, center)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    # frame_advance(frame, pts)
    # cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
    cv2.imshow('frame', cv2.resize(frame, [widthscale, heightscale]))

    #Quit window with "q"
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break



'''
TODO:
    - work when ball is moving towards camera (no absolute movement)
'''