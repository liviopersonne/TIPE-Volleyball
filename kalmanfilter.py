import cv2
import numpy as np


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.predicted = (0, 0)


    def add(self, coords):
        point = np.array([[np.float32(coords[0])], [np.float32(coords[1])]])
        self.kf.correct(point)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        self.predicted = (x, y)
        




if __name__ == "__main__":
    k = KalmanFilter()
    k.add((10, 10))
    print("1", k.predicted)
    k.add((20, 20))
    print(k.predicted)
    k.add((30, 30))
    print(k.predicted)
    k.add((40, 40))
    print("good", k.predicted)
    k.add((50, 50))
    print(k.predicted)
    k.add((60, 60))
    print(k.predicted)
    k.add((70, 68))
    print(k.predicted)