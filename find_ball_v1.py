import cv2, time
import numpy as np
from kalmanfilter import KalmanFilter

video = cv2.VideoCapture('data\\point.mp4')
scaledown = .7  #Valeur de réduction de taille
precedant = None
global N         #Number of points in memory
N = 5
global tracklist
tracklist = []  #Liste des trajectoires possibles par score décroissant

def dist(pa, pb):
    return np.sqrt((pb[0] - pa[0])**2 + (pb[1] - pa[1])**2)

def angle(pa, pb):
    return np.angle(complex(pb[0] - pa[0], pb[1] - pa[1]))

class track:
    def __init__(self, n):
        self.speeds = [None for i in range(n)]            #Last n points
        self.angles = [None for i in range(n)]            #Last n angles
        self.last_point = None   #Last known point
        self.search = [(0, 0), 0]   #Coords  + radius to search next
        self.score = 0              #Score of this particular track (probability of ball)
        self.check = False          #Check if it is updated during a frame
        self.n = n                  #Number of last
        self.kalman = KalmanFilter()  #Kalman filter (to predict next point)
        self.predicted = (0, 0)     #Predicted next point

    def __repr__(self):
        return f"<Track>: Last = {self.last_point} // Score = {self.score} // Search_dist = {self.search[1]}"

    def copy(self):
        n = track(self.n)
        n.speeds = self.speeds
        n.angles = self.angles
        n.last_point = self.last_point
        n.search = self.search
        n.score = self.score
        n.check = self.check
        n.n = self.n
        return n

    def dist(self, point):
        # return dist(self.search[0], point)
        return dist(self.predicted, point)

    def connect(self, point):
        return self.dist(point) <= self.search[1]

    def score(self, point):
        return NotImplementedError("Bonne chance pour ça")
        '''
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
        

    def idle(self):         #What to do if the track isn't updated on a frame
        self.score -= 1     #Add idle decreace
        self.search[1] *= 2 #Search in a larger zone
        self.check = False

        self.kalman.add(self.predicted)
        self.predicted = self.kalman.predict()

        if self.score > 0:  #Supprime une trajectoire si son score descend en dessous de 0
            return self

    def add_point(self, point):
        p = self.last_point
        if p:
            currspeed = dist(p, point)
            currangle = angle(p, point)
            # currscore = self.score(point)
        else:
            currspeed = 15      #Default speed check
            currangle = None
        currscore = 0
        self.speeds = self.speeds[1:] + [currspeed]
        self.angles = self.angles[1:] + [currangle]
        self.score += currscore
        self.last_point = point
        self.check = False  #Dont re-check adter the track is created
        self.search = [point, currspeed*2]  #Admet une accélération max détectable de *2 en 1 frame
        self.kalman.add(point)
        self.predicted = self.kalman.predict()
        return self


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
    check, frame = video.read()  #Read
    if not check:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Get gray image
    gray = cv2.GaussianBlur(gray,(31,31), 0)  #Gaussian size needs to be odd
    # gray = cv2.medianBlur(gray, 15)
    if precedant is None:
        precedant = gray

    delta = cv2.absdiff(precedant, gray) #Get difference
    


    LOWPASS = 20
    HIGHPASS = 255
    threshold = cv2.threshold(delta, LOWPASS, HIGHPASS, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None,iterations = 10)  #Get larger whites
    threshold = cv2.erode(threshold, None, iterations = 10)

    result = cv2.bitwise_and(frame, frame, mask=threshold)
    grayresult = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) #Get gray image



    contours, _ = cv2.findContours(grayresult.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blank = np.zeros([int(height), int(width), 3], dtype=np.uint8)
    pts = []

    for contour in contours:
        if cv2.contourArea(contour) > 30: #https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html
            # rect = cv2.boundingRect(contour)
            center, radius = cv2.minEnclosingCircle(contour)
            

            if radius > 5 and radius < 30:
                pts.append(center)
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                cv2.drawContours(blank, [contour], -1, (255, 255, 255), 2)
                cv2.circle(frame, tuple(map(int, center)), int(radius), (255, 0, 0), 2)
                cv2.circle(blank, tuple(map(int, center)), int(radius), (255, 0, 0), 2)

            cv2.putText(frame, str(int(radius)), tuple(map(int, center)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2) #Draw radius value
                
    frame_advance(frame, pts)

    # result_edges = cv2.Canny(grayresult, 200, 200)

    # blank = cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)


    # hough_circles = cv2.HoughCircles(blank, cv2.HOUGH_GRADIENT, 1, 10, param1=150, param2=70, minRadius=0, maxRadius=200)
    # circles = np.uint16(np.around(hough_circles))
    # for circle in circles[0, :]:
    #     cv2.circle(frame, (circle[0], circle[1]), circle[2], (255,255,255), 3)
    #     cv2.circle(frame, (circle[0], circle[1]), 1, (255,255,255), 3)


    cv2.imshow('frame', cv2.resize(frame, [widthscale, heightscale]))

    precedant = gray
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break



'''
TODO:
    - work when ball is moving towards camera (no absolute movement)
'''