import cv2
import numpy as np
from circle import Circle
from reward_functions import reward

# Distance entre 2 points (xa, ya) et (xb, yb)
def dist(pa, pb):
        return np.sqrt((pb[0] - pa[0])**2 + (pb[1] - pa[1])**2)

# cos de l'angle ABC (produit scalaire)
def absbend(pa, pb, pc):
    scal = (pb[0] - pa[0])*(pc[0] - pb[0]) + (pb[1] - pa[1])*(pc[1] - pb[1])
    r = dist(pa, pb)*dist(pb, pc)
    if r:
        return scal / r
    else:
        return 1

# Classe de trajectoire
global circle_coef, box_coef, score_decrement
circle_coef = 3     # Coefficient multiplicateur lors du test précis de connexion (% du rayon du dernier cercle)
box_coef = 0.2      # Coefficient multiplicateur lors du test large de connexion (% du côté de la zone testée)
score_decrement = 20  # Score à retirer à une track si elle n'est pas connectée

class Track:
    # Constructeur
    def __init__(self, c1):
        self.last_circle = c1       #Last known circle
        self.n = 1                  #Number of circles
        self.trajectory = [c1]      #Tracked circles
        self.score = c1.score       #Score of this particular track (probability of ball)
        self.predicted = c1.center  #Predicted next coordinates
        self.check = False          #Check if it is updated during a frame

    # Représentation  
    def __repr__(self):
        return f"<Track: last = {tuple(map(round, self.last_circle.center))}, {round(self.last_circle.radius)} // score = {round(self.score, 2)} // n = {self.n}>"

    # Dessine la trajectoire sur une frame
    def draw(self, img, color=(0, 0, 255), thickness=3, draw_score=False):
        for c in self.trajectory:
            cv2.circle(img, tuple(map(round, c.center)), round(c.radius), color, thickness)
        cv2.circle(img, tuple(map(round, self.predicted)), round(self.last_circle.radius), (255, 0, 0), thickness)
        if draw_score:
            cv2.putText(img, str(round(self.score,2)), (round(self.predicted[0] + 20), round(self.predicted[1] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    # Copie l'objet (pour le passage d'une frame à l'autre)
    def copy(self):
        n = Track(self.trajectory[0])
        n.n, n.score, n.predicted, n.check = self.n, self.score, self.predicted, self.check
        n.trajectory = self.trajectory.copy()
        return n

    # Distance entre un cercle c et la prédiction d'un track
    def dist(self, c):
        return dist(self.predicted, c.center)

    # (test précis) Vérifie si un cercle c vérifie la connexion au track (renvoie un booléen)
    def connect_circle(self, c):
        global circle_coef
        return self.dist(c) <= self.last_circle.radius*circle_coef

    # (test large) Vérifie si un rectangle xywh est dans la zone globale de connexion au track (renvoie un booléen)
    def connect_box(self, x,y,w,h):
        global box_coef
        xp, yp = self.predicted
        test_x = x-w*box_coef <= xp <= x+w*(1+box_coef)
        test_y = y-h*box_coef <= yp <= y+h*(1+box_coef)
        return test_x and test_y
    
    def draw_connect_box(self, img,x,y,w,h,color=(0, 0, 255),thickness=3):
        global box_coef
        cv2.rectangle(img, (round(x-w*box_coef),round(y-h*box_coef)), (round(x+w*(1+box_coef)),round(y+h*(1+box_coef))), color, thickness)

    # Critères de connexion     Valeurs possibles  Valeurs du filtre
      # er: Evolution du rayon     (0  -> +inf)    0    0    0.1  0.4
      # dp: Distance prédiction    (0  -> +inf)    0    0    0.3   2
      # sc: Score du cercle        (0  ->  1)     0.4  0.8    1    1
      # vp: Virage prédiction      (-1 ->  1)     -1   -1   -0.8 -0.6
    def connect_score(self, c):
        er = abs(c.radius - self.last_circle.radius) / self.last_circle.radius
        dp = self.dist(c) / self.last_circle.radius
        sc = c.score
        if self.n >= 2:
            vp = absbend(self.predicted, self.last_circle.center, c.center)
        
        score_er = reward(er, -1, 0, 0.1, 0.4)
        score_dp = reward(dp, 0, 0, 0.3, 2)
        score_sc = reward(sc, 0.4, 0.8, 1, 2)
        score_total = score_er + score_dp + score_sc
        if self.n >= 2:
            score_vp = reward(vp, -2, -1, -0.8, -0.6)
            score_total += score_vp

        return score_total

    # Met à jour la prédiction de cette trajectoire
    def update_prediction(self):
        if self.n > 1:            # On effectue la prédiction que la ballon va continuer en ligne droite sans variation de vitesse
            dx = self.last_circle.center[0] - self.trajectory[-2].center[0]
            dy = self.last_circle.center[1] - self.trajectory[-2].center[1]
            self.predicted = (self.last_circle.center[0] + dx, self.last_circle.center[1] + dy)
        else:
            self.predicted = self.last_circle.center

    # Ajoute un cercle c au bout de la trajectoire (cScore est le score de connexion calculé au préalable)
    def add_circle(self, c, cScore=None):
        if cScore is None:
            cScore = self.connect_score(c)  # On calcule le score
        self.score += cScore
        self.last_circle = c
        self.n += 1
        self.trajectory.append(c)
        self.update_prediction()
        
    # Comportement à prendre si la trajectoire n'est pas continuée
    def idle(self):
        global score_decrement
        if self.n > 1 and self.score >= score_decrement:
            new_track_1 = self.copy()  # Ajoute prediction
            new_track_1.add_circle(Circle(self.predicted, self.last_circle.radius, 1), -score_decrement)
            new_track_2 = self.copy()  # Ajoute immobile
            new_track_2.add_circle(self.last_circle, -score_decrement)
            return [new_track_1, new_track_2]
        else:
            return []
        


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