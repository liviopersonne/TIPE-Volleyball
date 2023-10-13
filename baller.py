import cv2, time, random
import numpy as np
from circle import Circle
import os

class Color:
    red = (0, 0, 255)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    orange = (0, 128, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 128)
    cyan = (255, 255, 0)
    purple = (255, 0, 127)
    pink = (255, 0, 255)
    darkpurple = (153, 0, 76)
    darkblue = (153, 0, 0)
    teal = (153, 153, 0)
    darkgreen = (0, 153, 0)
    gray = (127, 127, 127)
    black = (0, 0, 0)
    white = (255, 255, 255)

    colors = [red, blue, green, orange, yellow, green, cyan, purple, pink, darkpurple, darkblue, teal, darkgreen, gray, white]

    def random():
        i = random.randint(0, len(Color.colors)-1)
        return Color.colors[i]

def dist(pa, pb):
    return np.sqrt((pb[0] - pa[0])**2 + (pb[1] - pa[1])**2)

l = len(os.listdir("rawdata/edges"))
def edge(number):
    return cv2.imread(f"rawdata/edges/{number}.png", 1)

def argmin(l):
    (ind, m) = (0, l[0])
    for (i, v) in enumerate(l):
        if v < m:
            ind = i
            m = v
    return (ind, m)

def cont_dist(cont1, cont2, lim):
    [[x1, y1]] = cont1[0]
    [[z1, t1]] = cont1[-1]
    [[x2, y2]] = cont2[0]
    [[z2, t2]] = cont2[-1]

    (i, m) = argmin([
        dist((x1, y1), (x2, y2)),
        dist((x1, y1), (z2, t2)),
        dist((z1, t1), (x2, y2)),
        dist((z1, t1), (z2, t2))])

    if m < lim:
        return i
    else:
        return -1

# Connects contours which have close ends
def connect_contours(contours, lim):
    for (n1, c1) in enumerate(contours):
        for (n2, c2) in enumerate(contours[n1+1:]):
            i = cont_dist(c1, c2, lim)
            if i != -1:
                del(contours[n2])
                del(contours[n1])
                if i == 0:
                    contours.append(np.concatenate([np.array(list(reversed(c1))), c2]))
                elif i == 1:
                    contours.append(np.concatenate([c2, c1]))
                elif i == 1:
                    contours.append(np.concatenate([c1, c2]))
                elif i == 1:
                    contours.append(np.concatenate([c1, np.array(list(reversed(c2)))]))
                return connect_contours(contours, lim)
    return contours


def detect(edges):
    h, w, _ = np.shape(edges)
    lim = 3
    coef = 500//h
    gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10)
    

    print("Before :", len(contours))
    out = edges.copy()
    for cont in contours:
        cv2.drawContours(out, [cont], -1, Color.random())
    cv2.imshow("edges", cv2.resize(out, [w*coef, h*coef]))
    cv2.waitKey(0)

    # Connect close contours
    contours = tuple(connect_contours(list(contours), lim))

    print("After :", len(contours))
    out = edges.copy()
    for cont in contours:
        cv2.drawContours(out, [cont], -1, Color.random())
    cv2.imshow("edges", cv2.resize(out, [w*coef, h*coef]))
    cv2.waitKey(0)


detect(edge(33))