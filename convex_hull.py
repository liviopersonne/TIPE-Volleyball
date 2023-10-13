import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

'''
Formule du contour
[
    [[x1, y1]]
    [[x2, y2]]
    [[x3, y3]]
    ...
    [[xn, yn]]
]
Où chaque (xi, yi) (xi+1, yi+1) est une droite du contour.
'''

outfile = open("data/out.txt", "w")
img = cv2.imread("data/shape3.png", 1)
img = cv2.resize(img, [500, 500])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Trouve l'indice du point d'abscisse minimale dans un contour
def ind_abs_min(cont: np.array):
    m = cont[0]  #Point d'abscisse minimale
    ind = 0      #Indice d'abscisse minimal
    for i in range(len(cont)):
        [[x, y]] = cont[i]
        if(x < m[0][0]):  #Point plus gaiche trouvé
            m = [[x, y]]
            ind = i
    return ind

# Trouve l'angle entre 2 points
def angle(a, b):
    [[x1, y1]] = a
    [[x2, y2]] = b
    return np.angle(complex(x2-x1, y2-y1))

# Trouve l'orientation de x par rapport à la droite (ab) -> -1 pour gauche, 0 pour aligné, 1 pour droite
def orient(a, b, x):
    [[x1, y1]] = a
    [[x2, y2]] = b
    [[x3, y3]] = x
    z = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if z == 0:
        return 0
    elif z > 0:
        return -1
    else:
        return 1

# Trouve si un nombre est situé entre deux autres
def is_between(x, a, b):
    return a <= x and x <= b


def hull(cont: np.array, img):
    nb_pts, _, _ = cont.shape
    if(nb_pts < 3):
        return cont

    # Point le plus à gauche
    ind = ind_abs_min(cont)

    # Cherche l'orientation du contour
    j = 0
    while (j < len(cont) - 2):
        ori = orient(cont[j], cont[j+1], cont[j+2])
        if (ori != 0):
            break
        j += 1
    else:
        print("Ligne droite")
        return cont
    if (ori == 1):
        print("Gauche")
    elif (ori == 0):
        print("Milieu")
    else:
        print("Droite")


    output = np.zeros((nb_pts+1, 1, 2), int)
    output[0] = cont[ind]
    j = 0  #Indice du dernier point ajouté dans le hull

    #On boucle dans l'ordre en commençant et en terminant par le point le plus à gauche
    for i in list(range(ind+1, len(cont))) + list(range(ind+1)):
        cpy = img.copy()
        [[x, y]] = cont[i]
        #Tant qu'on a des angles sortants on enlève un point
        while (j > 0 and orient(output[j-1], output[j], cont[i]) != ori):
            j -= 1


            cv2.circle(cpy, (x, y), 10, (0, 0, 255), 2)
            cv2.circle(cpy, (output[j][0][0], output[j][0][1]), 10, (0, 0, 255), 2)
            cv2.drawContours(cpy, [output[:j+1]], -1, (0, 0, 255), 2)
            cv2.imshow("shape", cpy)
            if cv2.waitKey(waittime) & 0xFF == ord('q'):
                break


        j += 1
        output[j] = cont[i]


        cv2.circle(cpy, (x, y), 10, (200, 0, 100), 2)
        cv2.drawContours(cpy, [output[:j+1]], -1, (255, 0, 255), 2)
        cv2.imshow("shape", cpy)
        if cv2.waitKey(waittime) & 0xFF == ord('q'):
            break
    
    return output[:j]

# Transforme une liste de points en contour
def pts_to_contour(l):
    n = np.empty((len(l), 1, 2), int)
    for (i, (x, y)) in enumerate(l):
        n[i] = [[x, y]]
    return n

# Transforme un contour en une liste de points
def contour_to_pts(c):
    l = c.tolist()
    return list(map(lambda x: (x[0][0], x[0][1]), l))

# Plot une liste de points
def plot_pts(l):
    xlist = map(lambda x: x[0], a)
    ylist = map(lambda x: x[1], a)
    plt.plot(list(xlist), list(ylist), "ro")

global waittime
waittime = 0

for cont in contours:
    cpy = img.copy()
    h = hull(cont, img)
    cv2.drawContours(cpy, [h], -1, (255, 0, 255), 3)
    cv2.imshow("shape", cpy)
    cv2.waitKey(0)
    h = cv2.convexHull(cont)
    cv2.drawContours(cpy, [h], -1, (255, 0, 0), 3)
    cv2.imshow("shape", cpy)
    cv2.waitKey(0)




a = [(100, 100), (440, 100), (200, 220), (233, 300), (200, 230), (150, 400)]

img = np.zeros((500, 500, 3), np.uint8)
img[:,:] = (255,255,255)
c = pts_to_contour(a)
cv2.drawContours(img, c, -1, (255, 0, 255), 3)
h = hull(c, img)
cv2.drawContours(img, [h], -1, (255, 0, 255), 3)
cv2.imshow("shape", img)
cv2.waitKey(0)
h = cv2.convexHull(c)
cv2.drawContours(img, [h], -1, (255, 0, 0), 3)
cv2.imshow("shape", img)
cv2.waitKey(0)



# cv2.circle(img, (312, 740), 30, (200, 0, 100), 3)
outfile.close()
# cv2.imshow("shape", cv2.resize(img, [500, 500]))
# cv2.waitKey(0)