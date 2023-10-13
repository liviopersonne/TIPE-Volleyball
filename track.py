from kalmanfilter import KalmanFilter
from reward_functions import reward
from circle import Circle
import cv2
import numpy as np

#Distance between 2 points (xa, ya) and (xb, yb)
def dist(pa, pb):
        return np.sqrt((pb[0] - pa[0])**2 + (pb[1] - pa[1])**2)

#Angle formed by the line (pa pb)
def angle(pa, pb):
    return np.angle(complex(pb[0] - pa[0], pb[1] - pa[1]))

#cos de l'angle ABC (produit scalaire)
def absbend(pa, pb, pc):
    scal = (pb[0] - pa[0])*(pc[0] - pb[0]) + (pb[1] - pa[1])*(pc[1] - pb[1])
    r = dist(pa, pb)*dist(pb, pc)
    if r:
        scal /= r
    else:
        scal = 1
    return scal

#Classe de trajectoire
KALMANLIMIT = 99
class Track:
    #Constructeur
    def __init__(self, c1):
        self.last_circle = c1       #Last known circle
        self.n = 1                  #Number of circles
        self.trajectory = [c1]      #Tracked circles
        self.score = c1.area_ratio  #Score of this particular track (probability of ball)
        self.kalman = KalmanFilter()  #Kalman filter (to predict next point)
        self.kalman.add(c1.center)
        self.predicted = c1.center  #Predicted next coordinates
        self.check = False          #Check if it is updated during a frame

    #Représentation  
    def __repr__(self):
        return f"<Track: last = {tuple(map(round, self.last_circle.center))}, {round(self.last_circle.radius)} // score = {round(self.score, 2)} // n = {self.n}>"

    #Dessine la trajectoire sur une frame
    def draw(self, img, color=(0, 0, 255), thickness=3):
        for c in self.trajectory:
            cv2.circle(img, tuple(map(round, c.center)), round(c.radius), color, thickness)
        cv2.circle(img, tuple(map(round, self.predicted)), round(self.last_circle.radius), (255, 0, 0), thickness)

    #Copie l'objet (pour le passage d'une frame à l'autre)
    def copy(self):
        n = Track(self.trajectory[0])
        n.n = self.n
        n.trajectory = self.trajectory.copy()
        n.score = self.score
        n.kalman = self.kalman
        n.predicted = self.predicted
        n.check = self.check
        return n

    def dist(self, c):
        return dist(self.predicted, c.center)

    def connect_circle(self, c):
        return self.dist(c) <= self.last_circle.radius*3

    def connect_box(self, x,y,w,h):
        xp, yp = self.predicted
        return x-w/2 <= xp <= x+w*1.5 and y-h/2 <= yp <= y+h*1.5

    def connect_score(self, c):
        s1, s2, s3, s4, s5, s6 = [0 for i in range(6)]
        
        s1 = abs(c.radius - self.last_circle.radius) / self.last_circle.radius  #% d'evolution du rayon (0 -> +inf)
        s1 = reward(s1, -1, 0, 0.1, 0.4)
        s6 = reward(c.radius, 5, 6, 14, 18)  #rayon absolu
        
        s3 = c.area_ratio       #circle score (1 -> 0)
         
        if self.n >= 2:
            s4 = absbend(self.trajectory[-2].center, self.trajectory[-1].center, c.center)  #bend with last 2 points (1 -> -1)
            s4 = reward(s4, 0.6, 0.8, 1, 2)

        if self.n > 1:
            s2 = self.dist(c)       #distance to prediction (0 -> +inf)
            s2 = reward(s2, -1, 0, self.last_circle.radius/3, self.last_circle.radius*2)

            s5 = absbend(self.predicted, self.last_circle.center, c.center)     #bend with prediction (-1 -> 1)
            s5 = reward(s5, -2, -1, -0.8, -0.6)

        # print(round(s1,2), round(s2,2), round(s3,2), round(s4,2), round(s5,2), round(s6, 2))
        return s1 + s2 + s3 + s4 + s5 + s6

    def add_circle(self, c, cs=None):
        if cs is None:
            cs = self.connect_score(c)
        self.score += cs
        self.last_circle = c
        self.n += 1
        self.trajectory.append(c)
        self.kalman.add(c.center)
        if self.n >= KALMANLIMIT:
            self.predicted = self.kalman.predicted
        elif self.n > 1:
            ## Change this ##
            dx = c.center[0] - self.trajectory[-2].center[0]
            dy = c.center[1] - self.trajectory[-2].center[1]
            self.predicted = (c.center[0] + dx, c.center[1] + dy)
        else:
            self.predicted = c.center

    def idle(self):
        pass


if __name__ == '__main__':
    img = cv2.imread("data\\blue_background.webp")

    t = Track(Circle((10, 700), 20, 0.7))
    t.add_circle(Circle((30, 650), 20, 1))
    t.add_circle(Circle((50, 610), 20, 1))
    t.add_circle(Circle((60, 570), 20, 1))
    t.add_circle(Circle((90, 506), 20, 1))
    t.add_circle(Circle((100, 430), 20, 1))
    t.add_circle(Circle((130, 400), 20, 1))
    t.add_circle(Circle((150, 400), 20, 1))
    for i in range(10):
        t.add_circle(Circle(t.predicted, 20, 1))
    t.add_circle(Circle((450, 400), 20, 1))


    t.draw(img, (255, 100, 100))
    cv2.imshow('image', cv2.resize(img, [960, 540]))
    cv2.waitKey(0)