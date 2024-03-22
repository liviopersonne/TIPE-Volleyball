import numpy as np
import matplotlib.pyplot as plt


def reward(x, x1, x2, x3, x4):
    maxoutput = 1
    exponent = 2

    if x < x1:      #Under minimum
        return -((x1-x)/(x2-x1)*2)**exponent
    elif x < (x1+x2)/2:     #Lower than usual (left)
        return ((x-x1)/(x2-x1)*2)**exponent * maxoutput/2
    elif x < x2:            #Lower than usual (right)
        return (2 - ((x2-x)/(x2-x1)*2)**exponent) * maxoutput/2
    elif x < x3:    #Usual
        return maxoutput
    elif x < (x3+x4)/2:     #Higher than usual (left)
        return (2 - ((x-x3)/(x4-x3)*2)**exponent) * maxoutput/2
    elif x < x4:            #Higher than usual (right)
        return ((x4-x)/(x4-x3)*2)**exponent * maxoutput/2
    else:           #Over maximum
        return -((x-x4)/(x4-x3)*2)**exponent

if __name__ == '__main__':
    #This shows an axample of the function
    x1 = -1      #Minimum found
    x2 = 0      #Minimum usual
    x3 = 1      #Maximum usual
    x4 = 2      #Maximum found

    side = 1

    X = np.linspace(x1-side, x4+side, 10000)
    Y = [reward(x, x1, x2, x3, x4) for x in X]

    plt.plot(X, Y)
    M = reward(x2, x1, x2, x3, x4)
    m = min(reward(x1-side, x1, x2, x3, x4), reward(x4+side, x1, x2, x3, x4))
    plt.plot([x1, x1], [m, M], "r:")
    plt.plot([x2, x2], [m, M], "r:")
    plt.plot([x3, x3], [m, M], "r:")
    plt.plot([x4, x4], [m, M], "r:")
    plt.plot([x1-side, x4+side], [0, 0], "r")
    plt.title("Fonction de récompense")
    plt.text(x1, M, "Minimum", horizontalalignment='center')
    plt.text(x2, M, "Lower", horizontalalignment='center')
    plt.text(x3, M, "Higher", horizontalalignment='center')
    plt.text(x4, M, "Maximum", horizontalalignment='center')
    # plt.grid()
    plt.show()
