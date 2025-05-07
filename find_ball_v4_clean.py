import cv2, time
import numpy as np
from track import Track
from circle import Circle



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

    colors = [red, blue, green, orange, yellow, green, cyan, purple, pink, darkpurple, darkblue, teal, darkgreen, gray, black, white]

''' 
Score criteria:
    Big speed
    Top of screen
    Corresponding angle / position (KALMAN FILTER)
    Corresponding radius
    Color ?
    IS CIRCLE
        Height / Width ratio
        Compactness: Perimeter / Area
        Hough Detection
        Image comparison
'''


#Give a score to a contour regarding only its shape and position
'''
What is looked at:
    Height ratio (height comared to other contours)
        smaller = better
    Contour area ??
        usually between 300 and 800 (depends a lot on speed)
        minimum seen : 40
        maximim seen : 1200
    Height / Width ratio
        usually between 0.7 and 1.8
        minimum seen : 0.5
        maximum seen : 3.2
    Extent : Object area / Bounding rectangle area
        ususlly between 0.5 and 0.75
        minimum seen : 0.25
        maximum seen : 0.85
    Compactness: Perimeter / Area
        usually between 0.15 and 0.35
        minimum seen : 0.13
        maximum seen : 0.58
    Solidity : Contour Area / Convex Hull Area
        usually between 0.8 and 1.0
        minimum seen : 0.6
        maximum seen : 1.0

    Hough Detection
    Image comparison
    COCO image detection
'''

#Checks if an image contains a principal circular object
def findcircles(raw, x, y, height, width):
    img = cv2.GaussianBlur(raw, (5,5), 0)
    edges = cv2.Canny(img, 200, 200)
    circles = []
    
    # Hough circles method (bad)
    # hough_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=10, minRadius=0, maxRadius=200)
    # if hough_circles is not None:
    #     circles = np.uint16(np.around(hough_circles))
    #     for (x, y, r) in circles[0, :]:
    #         cv2.circle(cpy, (x, y), r, (255,0,0), 1)
    #         cv2.circle(cpy, (x, y), 0, (0,0,255), 1)

    #Find contours method
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        hull = cv2.convexHull(contour)
        center, radius = cv2.minEnclosingCircle(hull)

        hull_area = cv2.contourArea(hull)
        circle_area = np.pi*(radius**2)
        area_ratio = hull_area/circle_area        

        if area_ratio >= 0.6:
            circles.append(Circle((round(center[0] + x), round(center[1] + y)), radius, area_ratio))


        #Secondary info
        # hull_perimeter = cv2.arcLength(hull, True)
        # perimeter_ratio = hull_perimeter/hull_area

    return circles


def main(filename):
    ####################
    #### Constantes ####
    ####################
    video = cv2.VideoCapture(f'data/{filename}.mp4')
    scaledown = .7  #Valeur de réduction de taille
    precedant = None
    global tracklist
    tracklist = []  #Liste des trajectoires possibles par score décroissant


    ########################
    #### Get Properties ####
    ########################
    if video.isOpened():  
        vidwidth = int(video.get(3))
        vidwidthscale = int(vidwidth * scaledown)
        vidheight = int(video.get(4))
        vidheightscale = int(vidheight * scaledown)
        vidfps = video.get(5)
        vidframe_count = int(video.get(7))
        print(vidwidth, vidheight, vidfps, vidframe_count)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(f"data/output.mp4", fourcc, vidfps, (round(vidwidth),round(vidheight)))


    while video.isOpened():
        ######################################################
        #### Read frame and close at the end of the video ####
        ######################################################
        check, frame = video.read()
        if not check:
            break

        ##########################################
        #### Preprocessing (grayscale + blur) ####
        ##########################################
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Get gray image
        gray = cv2.GaussianBlur(gray,(31,31), 0)  #Gaussian size needs to be odd
        if precedant is None:
            precedant = gray

        delta = cv2.absdiff(precedant, gray) #Get difference
        precedant = gray

        ########################################
        #### Get contours of movement areas ####
        ########################################
        LOWPASS = 20
        HIGHPASS = 255
        threshold = cv2.threshold(delta, LOWPASS, HIGHPASS, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold,None,iterations = 10)  #Get larger whites
        threshold = cv2.erode(threshold, None, iterations = 10)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        ##########################
        #### Analyse contours ####
        ##########################
        newtracks = []
        for contour in contours:   #https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html
            ############################################
            #### Find circles in each movement area ####
            ############################################
            x,y,width,height = cv2.boundingRect(contour)
            zone = frame[y:y+height, x:x+width]
            circles = findcircles(zone, x, y, height, width)

            #####################################
            #### Add circles to trajectories ####
            #####################################
            for t in tracklist:
                if t.connect_box(x,y,width,height):  #1st fast check
                    for c in circles:
                        cs = t.connect_score(c)
                        if cs > 0:          #2nd slow but precise check
                            n = t.copy()
                            n.add_circle(c,cs)
                            newtracks.append(n)
                            t.check = True
                            circles.remove(c)


            ########################################################
            #### Create new trajectories with unmatched circles ####
            ########################################################
            for c in circles:
                newtracks.append(Track(c))

        #####################################################
        #### Vérifier les zones d'intérêt sans mouvement ####
        #####################################################        
        for t in tracklist:
            if not t.check:
                xp, yp = t.predicted
                if 0 <= xp <= vidwidth and 0 <= yp <= vidheight:
                    buffer = round(t.last_circle.radius*3)
                    zone = frame[max(0, yp-buffer):min(vidheight,yp+buffer), max(0, xp-buffer):min(vidwidth,xp+buffer)]
                    circles = findcircles(zone, xp-buffer, yp-buffer, 2*buffer, 2*buffer)

                    for c in circles:
                        cs = t.connect_score(c)
                        if cs > 0:
                            n = t.copy()
                            n.add_circle(c,cs)
                            newtracks.append(n)
                            t.check = True

        ###############################################
        #### Traîter les trajectoires non touchées ####
        ###############################################
        for t in tracklist:
            if not t.check:
                if t.score >= 20:
                    t.add_circle(Circle(t.predicted, t.last_circle.radius, 1), -40)
                    newtracks.append(t)

        ################################
        #### Afficher les résultats ####
        ################################
        tracklist = newtracks

        #Show all found tracks
        for t in tracklist:
            t.draw(frame, Color.red, -1)

        # Show best found track
        if tracklist:
            max(tracklist, key = lambda x: x.score).draw(frame, Color.darkgreen, -1)

        cv2.imshow('frame', cv2.resize(frame, [vidwidthscale, vidheightscale]))
        out.write(frame)


        #Quit window with "q"
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    out.release()


start = time.time()
main("videos/1_belle_recup.mp4")
print(round(time.time() - start, 2))


'''
TODO:
    - work when ball is moving towards camera (no absolute movement)
    - not lose trajectories
    - deal when prediction is outside box
    - ball over net
'''