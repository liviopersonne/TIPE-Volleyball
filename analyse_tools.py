import numpy as np
import cv2
from color import Color
import matplotlib.pyplot as plt

# Calcule l'enveloppe convexe en renfermant des points autour des pixels blancs
def hull_delta_close(delta, height, width, LUM_BAS=30, precision=1, pts=50):
    precision = 1
    assert(0 < precision and precision <= 1)
    contour = np.empty((pts, 1, 2), int)
    pas_tour = 2*(width+height) / pts
    a = 0
    for i in range(pts): #Crée le rectangle de base
        if (a <= width): #Haut de l'écran
            x = a
            y = 0
        elif (a <= width+height): #Droite de l'écran
            x = width
            y = a - width
        elif (a <= 2*width+height): #Bas de l'écran
            x = width - (a - (width + height))
            y = height
        else: #Gauche de l'écran
            x = 0
            y = height - (a - (2*width + height))
        contour[i] = [[x, y]]
        a += pas_tour

    move_pts = list(range(pts)) #Indice des points qu'il faut continuer à bouger
    (cx, cy) = (width//2, height//2) #Centre du rectangle
    coef = max(cx, cy) * precision
    thresh = cv2.threshold(delta, LUM_BAS, 255, cv2.THRESH_BINARY)[1]



    for i in range(pts):  #Rapproche les points un à un
        prec_lum = 0 #Valeur precedente de lum
        [[x, y]] = contour[i]
        (dx, dy) = ((cx-x), (cy-y))
        (dx, dy) = (dx/coef, dy/coef)

        (distx, disty) = ((cx-x), (cy-y))
        while abs(distx) > 3 or abs(disty) > 3:
            (nx, ny) = (x+dx, y+dy)
            (x, y) = (nx, ny)
            lum = int(delta[int(y)][int(x)])
            if (lum > LUM_BAS):
                break

            prec_lum = lum
            contour[i] = [[round(x), round(y)]]
            (distx, disty) = ((cx-x), (cy-y))

        # colordelta = cv2.cvtColor(delta, cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(colordelta, contour, -1, Color.red, 5)
        # cv2.circle(colordelta, (cx, cy), 2, Color.blue, -1)
        # cv2.imshow("delta", colordelta)
        # cv2.waitKey(300)
        # cv2.destroyWindow("delta")



    # plt.plot(*contour_to_cart(contour))
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.plot(*contour_to_polar(height, width, contour), "ro")


    # plt.show()

    return contour






# Fonctions de convertions contour / pts / coordonnées
def cart_to_polar(height, width, x, y):
    assert(len(x) == len(y))
    hpi = np.pi / 4
    (cx, cy) = (width//2, height//2) #Centre du rectangle
    rho, phi = ([0] * len(x), [0] * len(x))
    for i in range(len(x)):
        rho[i] = np.sqrt((x[i]-cx)**2 + (y[i]-cy)**2)
        phi[i] = np.arctan2((y[i]-cy), (x[i]-cx)) + hpi
    return(rho, phi)

def polar_to_cart(height, width, rho, phi):
    assert(len(rho) == len(phi))
    hpi = np.pi / 4
    (cx, cy) = (width//2, height//2) #Centre du rectangle
    x, y = ([0] * len(rho), [0] * len(rho))
    for i in range(len(rho)):
        x[i] = rho[i] * np.cos(phi[i] - hpi) + cx
        y[i] = rho[i] * np.sin(phi[i] - hpi) + cy
    return(x, y)

def contour_to_cart(contour):
    x, y = ([], [])
    for i in contour:
        [[a, b]] = i
        x.append(a)
        y.append(b)
    return(x, y)

def cart_to_contour(x, y):
    assert(len(x) == len(y))
    contour = np.empty((len(x), 1, 2), int)
    for i in range(len(x)):
        contour[i] = [[x[i], y[i]]]
    return contour

def contour_to_polar(height, width, contour):
    return(cart_to_polar(height, width, *contour_to_cart(contour)))

def polar_to_contour(height, width, rho, phi):
    return(cart_to_contour(*polar_to_cart(height, width, rho, phi)))






if __name__ == "__main__":
    x = [154, 155, 155, 161, 169, 179, 193, 211, 234, 263, 289, 295, 189, 317, 329, 337, 333, 340, 334, 298, 189, 189, 189, 189, 324, 318, 341, 308, 269, 216, 204, 180, 186, 172, 170, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 143, 145, 147, 150, 151]
    y = [190, 183, 170, 161, 152, 141, 118, 111, 96, 77, 70, 93, 228, 113, 126, 148, 176, 201, 228, 249, 232, 232, 233, 233, 374, 393, 462, 462, 450, 357, 431, 356, 234, 273, 266, 234, 234, 234, 233, 233, 232, 232, 231, 231, 231, 216, 209, 203, 198, 193]


    plt.plot(x, y)
    plt.grid()
    plt.show()


    rho, phi = cart_to_polar(300, 300, x, y)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(phi, rho)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.grid(True)
    plt.show()

    x2, y2 = polar_to_cart(300, 300, rho, phi)
    plt.plot(x2, y2)
    plt.grid()
    plt.show()


























