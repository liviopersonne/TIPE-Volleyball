#Circle class
class Circle:
    def __init__(self, center, radius, score):
        self.center = center        #Coords of center
        self.radius = radius        #Radius
        self.score = score          #Score calculated by the kd tree

    def __repr__(self):
        return f"<Circle: {tuple(map(round, self.center))} // r = {self.radius} // s = {self.score}>"