import cv2, time
import numpy as np
from track import Track
from circle import Circle
import os
from analyse_tools import *
import classifieur
from color import Color
import csv


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
# TODO(A retirer)
def contourScore(contour, miny, maxy, meany):  #https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c
    area = cv2.contourArea(contour)
    permieter = cv2.arcLength(contour, True)
    x,y,width,height = cv2.boundingRect(contour)
    center,radius = cv2.minEnclosingCircle(contour)
    hull = cv2.convexHull(contour)       #Convex shape that contains everything
    hull_area = cv2.contourArea(hull)

    aspect_ratio = height / width
    score1 = reward(aspect_ratio, 0.5, 0.7, 1.8, 3.2)
    extent = area / (width*height)
    score2 = reward(extent, 0.25, 0.5, 0.75, 0.85)
    if area:
        compactness = permieter / area
    else:
        compactness = 0
    score3 = reward(compactness, 0.13, 0.15, 0.35, 0.58)
    if hull_area:
        solidity = area / hull_area
    else:
        solidity = 0
    score4 = reward(solidity, 0.6, 0.8, 1, 2)
    heightratio = (center[1] - miny) / (meany - miny + 1)  #"+1" to avoid 0 division
    score5 = -heightratio

    totalscore = score1 + score2 + score3 + score4 + score5

    return totalscore

# TODO(A retirer)
#Checks if an image contains a principal circular object
def findcircles(raw, x, y, height, width, infos=False):
    img = cv2.GaussianBlur(raw, (5,5), 0)
    edges = cv2.Canny(img, 200, 200)
    circles = []

    
    if infos:
        cv2.imshow("Raw", cv2.resize(raw, [width*6, height*6]))
        cv2.imshow("Canny 1", cv2.resize(edges, [width*6, height*6]))
        cv2.imshow("Canny 2", cv2.resize(cv2.Canny(img, 200, 20), [width*6, height*6]))
        cv2.imshow("Canny 3", cv2.resize(cv2.Canny(img, 5, 200), [width*6, height*6]))
        cv2.waitKey(0)
        cv2.destroyWindow("Raw")
        cv2.destroyWindow("Canny 1")
        cv2.destroyWindow("Canny 2")
        cv2.destroyWindow("Canny 3")
        cpy = raw.copy()
        cpy2 = img.copy()
        cpy3 = raw.copy()
        cpy4 = raw.copy()
        blank1 = np.zeros([int(height), int(width), 3], dtype=np.uint8)
        blank2 = blank1.copy()

    
    # Hough circles method (bad)
    # hough_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=10, minRadius=0, maxRadius=200)
    # if hough_circles is not None:
    #     circles = np.uint16(np.around(hough_circles))
    #     for (x, y, r) in circles[0, :]:
    #         cv2.circle(cpy, (x, y), r, (255,0,0), 1)
    #         cv2.circle(cpy, (x, y), 0, (0,0,255), 1)

    #Find contours method
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(cpy4, contours, -1, (255, 0, 0), -1)
    for contour in contours:
        hull = cv2.convexHull(contour)
        center, radius = cv2.minEnclosingCircle(hull)

        hull_area = cv2.contourArea(hull)
        circle_area = np.pi*(radius**2)
        area_ratio = hull_area/circle_area
        


        if infos:
            cv2.drawContours(blank1, [contour], -1, (255, 0, 0), 0)
            cv2.drawContours(blank1, [hull], -1, Color.red, 0)
            cv2.drawContours(blank2, [contour], -1, (255, 0, 0), 0)
            cv2.circle(blank2, (round(center[0]), round(center[1])), round(radius), Color.green, 0)
            print("Hull :", hull_area, " Circle :", circle_area, " Ratio :", area_ratio)
        

        if area_ratio >= 0.6:
            circles.append(Circle((round(center[0] + x), round(center[1] + y)), radius, area_ratio))


        #Secondary info
        # hull_perimeter = cv2.arcLength(hull, True)
        # perimeter_ratio = hull_perimeter/hull_area

        # print(round(area_ratio, 2), end = " ")

        if infos:
            if area_ratio >= 0.9:
                cv2.circle(cpy3, tuple(map(int, center)), int(radius), (0, 255, 0), 1)
            elif area_ratio >= 0.8:
                cv2.circle(cpy3, tuple(map(int, center)), int(radius), (0, 200, 100), 1)
            elif area_ratio >= 0.7:
                cv2.circle(cpy3, tuple(map(int, center)), int(radius), (0, 100, 200), 1)
            elif area_ratio >= 0.6:
                cv2.circle(cpy3, tuple(map(int, center)), int(radius), (0, 50, 255), 1)

    # print("")  #Retour à la ligne
    # l = len(os.listdir("rawdata/raw"))
    # cv2.imwrite(f"rawdata/raw/{l}.png", cpy)
    # cv2.imwrite(f"rawdata/edges/{l}.png", edges)
    if infos:
        cv2.imshow("1 zone", cv2.resize(cpy, [width*6, height*6]))
        cv2.imshow("2 flou", cv2.resize(cpy2, [width*6, height*6]))
        cv2.imshow("3 bords", cv2.resize(edges, [width*6, height*6]))
        cv2.imshow("4 contours", cv2.resize(cpy4, [width*6, height*6]))
        cv2.imshow("hull", cv2.resize(blank1, [width*6, height*6]))
        cv2.imshow("enc circle", cv2.resize(blank2, [width*6, height*6]))
        cv2.imshow("5 cercles", cv2.resize(cpy3, [width*6, height*6]))
        # for c in contours:
        #     nf = cpy.copy()
        #     cv2.drawContours(nf, [c], -1, (255, 0, 0), -1)
        #     cv2.imshow("Contour", cv2.resize(nf, [width*6, height*6]))
        #     cv2.waitKey(0)
        #     cv2.destroyWindow("Contour")
        k = cv2.waitKey(0) & 0xFF
        if k == 13:   #Skip rest of zones
            cv2.destroyWindow("1 zone")
            cv2.destroyWindow("2 flou")
            cv2.destroyWindow("3 bords")
            cv2.destroyWindow("4 contours")
            cv2.destroyWindow("5 cercles")
            cv2.destroyWindow("hull")
            cv2.destroyWindow("enc circle")
            return circles
        elif k == ord('e'):   #Export this image
            l = len(os.listdir('data\\'))
            cv2.imwrite(f"data\\img{l}_raw.png", cpy)
            cv2.imwrite(f"data\\img{l}_blur.png", cpy2)
            cv2.imwrite(f"data\\img{l}_edge.png", edges)
            cv2.imwrite(f"data\\img{l}_contours.png", cpy4)
            cv2.imwrite(f"data\\img{l}_hull.png", blank1)
            cv2.imwrite(f"data\\img{l}_encl_circle.png", blank2)
            cv2.imwrite(f"data\\img{l}_circles.png", cpy3)
        cv2.destroyWindow("1 zone")
        cv2.destroyWindow("2 flou")
        cv2.destroyWindow("3 bords")
        cv2.destroyWindow("4 contours")
        cv2.destroyWindow("5 cercles")
        try:
            cv2.destroyWindow("hull")
            cv2.destroyWindow("enc circle")
        except Exception:
            pass

    return circles

# Va faire une analyse d'une zone de l'image en utilisant le delta
def analyse(raw, delta, x, y, height, width, infos=False):
    LUM_BAS = 30
    thresh = cv2.threshold(delta, LUM_BAS, 255, cv2.THRESH_BINARY)[1]
    colorthresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    frame_area = height*width   #Aire de la zone
    white_pixels = cv2.countNonZero(thresh)    #Aire des pixels blancs dans la zone

    # Méthode du reserrement
    PRECISION, PTS = 1, 50
    hull = hull_delta_close(delta, height, width, LUM_BAS, PRECISION, PTS)  #Calcul de l'enveloppe
    cv2.drawContours(colorthresh, [hull], -1, Color.red, 2)
    hull_area = cv2.contourArea(hull)   #Aire de l'enveloppe
    center, radius = cv2.minEnclosingCircle(hull)  #Cercle circonscript à l'enveloppe
    circle_area = np.pi*(radius**2)   #Aire du cercle circonscript

    # print("White:", white_pixels, "Hull:", hull_area, "Circle:", circle_area, "Frame:", frame_area)
    solidite, circularite, taille = white_pixels/hull_area, hull_area/circle_area, white_pixels/frame_area
    print("Solidite:", solidite, "Circularite:", circularite, "Taille:", taille)

    ### VARIABLES ###
    csv_filename = "all.csv"
    NBR_VOISINS = 10
    data = {
        "solidite": solidite, "circularite": circularite, "taille": taille, 
        "white": white_pixels, "hull": hull_area, "circle": circle_area, "frame": frame_area
        }

    # Calcul du score de la forme selon l'arbre kd
    global kd_tree, kd_tree_criteria
    point_data = [data[crit] for crit in kd_tree_criteria]
    x = classifieur.Data.create_point(point_data)
    score_balle = classifieur.score_ppv(kd_tree, x, NBR_VOISINS)
    print("ppv", classifieur.k_ppv(kd_tree, x, NBR_VOISINS))
    print("Score ppv", score_balle)


    # 1: Solidité + vitesse de la forme
    # 2: Circularité de la forme
    # 3: Taille de la forme dans la frame
    # Adaboost // Kppv (arbre kd) // ID3

    cv2.imshow("analysed frame", raw)
    cv2.imshow("delta", delta)
    cv2.imshow("contour", colorthresh)

    key = cv2.waitKey(0) & 0xFF
    ball_test = None
    if key == 13: #Enter
        ball_test = True
    elif key == 8: #Backspace
        ball_test = False
    
    # Exportation dans le fichier csv
    if ball_test is not None:
        csv_data = {"id": -1, "positif": ball_test}|data
        print(csv_data)
        export_data(csv_filename, thresh, csv_data)

    cv2.destroyWindow("analysed frame")
    cv2.destroyWindow("delta")
    cv2.destroyWindow("contour")


# Exporte une image de seuil dans le fichier csv de nom filename
# avec les informations de solidité, circularité, et taille et si c'est le ballon
def export_data(filename, thresh, data):
    print(data)
    l = len(os.listdir('data/img'))
    cv2.imwrite(f"data/img/{l}.png", thresh)
    data["id"] = l
    path = f"data/{filename}"
    if os.path.isfile(path):
        csvfile = open(path, "a", newline="")
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=list(data.keys()))
    else:
        csvfile = open(path, "w", newline="")
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=list(data.keys()))
        writer.writeheader()
    writer.writerow(data)
    csvfile.close()
        


def main(filename, scaledown, frame_offset):
    ####################
    #### Constantes ####
    ####################
    video = cv2.VideoCapture(f'data\\{filename}.mp4')
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
        precedant, avg_frame, frames_seen = None, None, 0
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(f"data/output.mp4", fourcc, vidfps, (round(vidwidth),round(vidheight)))

    
    # Skip les premières frames
    for i in range(frame_offset):
        check, frame = video.read()
        if not check:
            break

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
        if frames_seen == 0:
            precedant = gray
            avg_frame = gray
        frames_seen += 1

        delta = cv2.absdiff(precedant, gray) # Différence avec l'image précédente
        avg_delta = cv2.absdiff(avg_frame, gray) # Différence avec l'image moyenne
        precedant = gray

        ########################################
        #### Get contours of movement areas ####
        ########################################
        LOWPASS = 10
        HIGHPASS = 255
        threshold_prec= cv2.threshold(delta, LOWPASS, HIGHPASS, cv2.THRESH_BINARY)[1]
        threshold_prec = cv2.dilate(threshold_prec,None,iterations = 10)  #Get larger whites
        low_threshold = cv2.bitwise_not(cv2.dilate(threshold_prec,None,iterations = 10))  # For the average image
        threshold_prec = cv2.erode(threshold_prec, None, iterations = 10)

        threshold_avg= cv2.threshold(delta, LOWPASS, HIGHPASS, cv2.THRESH_BINARY)[1]
        threshold_avg = cv2.dilate(threshold_avg,None,iterations = 10)  #Get larger whites
        threshold_avg = cv2.erode(threshold_avg, None, iterations = 10)

        threshold = cv2.bitwise_and(threshold_avg, threshold_prec)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Update average image
        still_gray = cv2.bitwise_and(gray, low_threshold)
        missing = cv2.bitwise_and(avg_frame, cv2.bitwise_not(low_threshold))
        gray_no_movement = cv2.addWeighted(still_gray, 1, missing, 1, 0)
        alpha = 2.5/(frames_seen)
        avg_frame = cv2.addWeighted(gray_no_movement, alpha, avg_frame, 1-alpha, 0.0)
        # Other version
        # alpha = 2.5/(frames_seen)
        # avg_frame = cv2.addWeighted(gray, alpha, avg_frame, 1-alpha, 0.0)



        # blank = np.zeros([int(vidheight), int(vidwidth), 3], dtype=np.uint8)
        # cont1 = blank.copy()
        # cont2 = blank.copy()
        # cv2.drawContours(cont1, contours, -1, Color.orange, 2)
        # cpy = frame.copy()
        # circles = findcircles(frame, 0, 0, vidheight, vidwidth, infos=False)
        # for c in circles:
        #     cv2.circle(cpy, c.center, round(c.radius), Color.green, 3)

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

            if width > 60 and height > 60:
                zonedelta = delta[y:y+height, x:x+width]
                analyse(zone, zonedelta, x, y, height, width, infos=INFOS)

            # circles = findcircles(zone, x, y, height, width, infos=INFOS)
            circles = []
            # cv2.rectangle(cont2, (x, y), (x+width, y+height), Color.red, 5)


            #####################################
            #### Add circles to trajectories ####
            #####################################
            for t in tracklist:
                if t.connect_box(x,y,width,height):  #1st fast check

                    # nf = frame.copy()
                    # cv2.rectangle(nf, (x,y), (x+width,y+height), Color.green, 3)
                    # t.draw(nf, Color.green, 3)
                    # cv2.imshow('Box connect', cv2.resize(nf, [vidwidthscale, vidheightscale]))
                    for c in circles:
                        # nf = frame.copy()
                        # for o in circles:
                        #     cv2.circle(nf, o.center, round(o.radius), Color.white, 3)
                        # t.draw(nf, Color.purple, -1)

                        cs = t.connect_score(c)
                        if cs > 0:          #2nd slow but precise check
                            color = Color.green

                            n = t.copy()
                            n.add_circle(c,cs)
                            newtracks.append(n)
                            t.check = True
                            circles.remove(c)
                        else:
                            color = Color.red
                        
                    #     cv2.circle(nf, c.center, round(c.radius), color, 3)
                    #     cv2.putText(nf, str(round(cs,2)), c.center, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
                    #     cv2.imshow('Circle connect', cv2.resize(nf[max(0, y-2*height):min(vidheight,y+2*height), max(0, x-2*width):min(vidwidth,x+2*width)], [vidwidthscale, vidheightscale]))
                    #     cv2.imshow('Circle connect', cv2.resize(nf, [vidwidthscale, vidheightscale]))
                    #     if cv2.waitKey(0) & 0xFF == ord('q'):
                    #         break
                    # if circles:
                    #     cv2.destroyWindow('Circle connect')
                else:
                    pass
                #     nf = frame.copy()
                #     cv2.rectangle(nf, (round(x-width/2),round(y-height/2)), (round(x+width*1.5),round(y+height*1.5)), Color.red, 3)
                #     t.draw(nf, Color.red, 3)
                #     cv2.imshow('Box connect', cv2.resize(nf, [vidwidthscale, vidheightscale]))
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break


            ########################################################
            #### Create new trajectories with unmatched circles ####
            ########################################################
            for c in circles:
                newtracks.append(Track(c))

        #####################################################
        #### Vérifier les zones d'intérêt sans mouvement ####
        #####################################################        
        for t in tracklist:
            print("Completing", t)
            if not t.check:
                nf = frame.copy()
                t.draw(nf)
                # cv2.imshow('Not connected tracks', cv2.resize(nf, [vidwidthscale, vidheightscale]))
                cv2.waitKey(0)
                cv2.destroyWindow('Not connected tracks')



                xp, yp = t.predicted
                if 0 <= xp <= vidwidth and 0 <= yp <= vidheight:
                    buffer = round(t.last_circle.radius*3)
                    zone = frame[max(0, yp-buffer):min(vidheight,yp+buffer), max(0, xp-buffer):min(vidwidth,xp+buffer)]
                    circles = findcircles(zone, xp-buffer, yp-buffer, 2*buffer, 2*buffer, infos=INFOS)

                    for c in circles:
                        nf = frame.copy()
                        for o in circles:
                            cv2.circle(nf, o.center, round(o.radius), Color.white, 3)
                        t.draw(nf, Color.purple, 3)

                        cs = t.connect_score(c)
                        if cs > 0:
                            color = Color.green

                            n = t.copy()
                            n.add_circle(c,cs)
                            newtracks.append(n)
                            t.check = True
                        else:
                            color = Color.red

                        cv2.circle(nf, c.center, round(c.radius), color, 3)
                        cv2.putText(nf, str(round(cs,2)), c.center, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
                        # cv2.imshow('No movement check', cv2.resize(nf[max(0, yp-2*buffer):min(vidheight,yp+2*buffer), max(0, xp-2*buffer):min(vidwidth,xp+2*buffer)], [vidwidthscale, vidheightscale]))
                        # cv2.imshow('No movement check', cv2.resize(nf, [vidwidthscale, vidheightscale]))
                        # if cv2.waitKey(0) & 0xFF == ord('q'):
                        #     break
                    # if circles:
                    #     cv2.destroyWindow('No movement check')

        ###############################################
        #### Traîter les trajectoires non touchées ####
        ###############################################
        for t in tracklist:
            if not t.check:
                if t.score >= 20:
                    print("Failsafe")
                    t.add_circle(Circle(t.predicted, t.last_circle.radius, 1), -40)
                    newtracks.append(t)


        tracklist = newtracks
        for t in tracklist:
            nf = frame.copy()
            t.draw(nf, Color.darkgreen, -1)
            cv2.imshow('track', cv2.resize(nf, [vidwidthscale, vidheightscale]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        if cv2.getWindowProperty('track', cv2.WND_PROP_VISIBLE) == 1:  #Si la fenêtre n'est pas fermée
            cv2.destroyWindow('track')



        # cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
        # cv2.imshow('midframe', cv2.resize(avg_frame, [vidwidthscale, vidheightscale]))
        # cv2.imshow('still gray', cv2.resize(still_gray, [vidwidthscale, vidheightscale]))
        # cv2.imshow('missing', cv2.resize(missing, [vidwidthscale, vidheightscale]))
        cv2.imshow('threshold', cv2.resize(threshold, [vidwidthscale, vidheightscale]))
        cv2.imshow('frame', cv2.resize(frame, [vidwidthscale, vidheightscale]))
        # cv2.imshow('circles', cv2.resize(cpy, [vidwidthscale, vidheightscale]))
        # out.write(cpy)
        # cv2.imshow('delta', cv2.resize(delta, [vidwidthscale, vidheightscale]))
        # cv2.imshow('threshold 1', cv2.resize(t1, [vidwidthscale, vidheightscale]))
        # cv2.imshow('threshold 2', cv2.resize(threshold, [vidwidthscale, vidheightscale]))
        # cv2.imshow('contours 1', cv2.resize(cont1, [vidwidthscale, vidheightscale]))
        # cv2.imshow('contours 2', cv2.resize(cont2, [vidwidthscale, vidheightscale]))

        #Quit window with "q"
        if cv2.waitKey(0) & 0xFF == ord('q'):
            # out.release()
            break


INFOS = False
frame_offset = 0
global kd_tree, kd_tree_criteria
kd_tree_criteria = ["solidite", "white", "frame"]
kd_tree = classifieur.build_kd_tree("data/all.csv", kd_tree_criteria)
main('videos\\4_service', .25, frame_offset)




# try:
#     out.release()
# except Exception:
#     pass

# im = cv2.imread("data\\faux_negatif.png")
# findcircles(im, 0, 0, im.shape[0], im.shape[1], infos=True)


'''
TODO:
    - work when ball is moving towards camera (no absolute movement)
    - not lose trajectories
    - deal when prediction is outside box
        File "c:\\Users\\livio\\Desktop\\fichiers_python\\fichiers\Volleyball\\find_ball_v4.py", line 345, in <module>
        zone = frame[max(0, yp-buffer):min(vidheight,yp+buffer), max(0, xp-buffer):min(vidwidth,xp+buffer)]
        TypeError: slice indices must be integers or None or have an __index__ method
    - ball over net
'''