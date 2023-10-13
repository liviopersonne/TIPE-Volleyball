#Circle class
class Circle:
    def __init__(self, center, radius, score):
        self.center = center        #Coords of center
        self.radius = radius        #Radius
        self.area_ratio = score     #Ratio hull_area/circle_area (score)

    def __repr__(self):
        return f"<Circle: {tuple(map(round, self.center))} // r = {self.radius} // s = {self.area_ratio}>"