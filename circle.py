import cv2

# Classe de cercle
class Circle:
    def __init__(self, center, radius, score):
        self.center = center        # Coordonnées du centre
        self.radius = radius        # Rayon
        self.score = score          # Score calculé par l'arbre kd

    # Dessine le cercle sur une image
    def draw(self, img, color=(0, 0, 255), thickness=3):
        cv2.circle(img, tuple(map(round, self.center)), round(self.radius), color, thickness)
    
    # Affichage d'un cercle dans le terminal
    def __repr__(self):
        return f"<Circle: {tuple(map(round, self.center))} // r = {self.radius} // s = {self.score}>"