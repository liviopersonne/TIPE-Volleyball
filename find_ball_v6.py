import cv2, time
import csv
import numpy as np
import os
from color import Color
from circle import Circle
import classifieur
from track import Track


# Trouve toutes les informations sur un contour unique - exporte et importe dans l'arbre kd et renvoie le cercle trouvé
def analyse_contour(full, contour, thresh, x, y, width, height, nbrFormes, infos=False):
    contourArea = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)  # Enveloppe convexe
    hullArea = cv2.contourArea(hull)
    (cx, cy), r = cv2.minEnclosingCircle(contour) # Cercle minimum
    circleArea = np.pi*(r**2)
    frameArea = width * height

    # Si l'une des aires est nulle alors il n'y a pas de cercle
    if min(contourArea, hullArea, circleArea, frameArea) == 0:
        return None

    solidite = contourArea / hullArea   # Mesure de la convexité d'un contour
    circularite = hullArea / circleArea # Mesure de la "circularité" d'un contour
    taille = circleArea / frameArea     # Mesure de la taille d'un contour dans son contexte

    # Donnée pour l'arbre kd
    data = {
        "solidite": solidite, "circularite": circularite, "taille": taille, 
        "area": contourArea, "hull": hullArea, "circle": circleArea,
        "frame": frameArea, "formes": nbrFormes
        }

    # Noter le contour avec l'arbre kd
    global kd_tree, kd_tree_criteria, kd_tree_normalization, nbr_voisins
    if kd_tree is not None:
        point_data = [data[crit] for crit in kd_tree_criteria] # Point dans l'espace de l'arbre kd
        point_classifieur = classifieur.Data.create_point(point_data)
        score_balle = classifieur.score_ppv(kd_tree, point_classifieur, nbr_voisins, kd_tree_normalization)
        if infos:
            print("ppv", classifieur.k_ppv(kd_tree, point_classifieur, 5, kd_tree_normalization))
            print("Score ppv", score_balle)
        return_info = Circle((cx+x, cy+y), r, score_balle)
    else: # Si l'arbre n'est pas défini on note le contour juste à l'aide de la circularité
        return_info = Circle((cx+x, cy+y), r, circularite)

    # Exporter la donnée dans l'arbre kd
    if infos:
        showImage = cv2.rectangle(full, (x, y), (x+width, y+height), Color.red, 2)
        cv2.imshow("analysed contour", showImage)
        key = cv2.waitKey(0) & 0xFF
        ball_test = None
        if key == 13: #Enter
            ball_test = True
        elif key == 8: #Backspace
            ball_test = False
        cv2.destroyWindow("analysed contour")

        global csv_filename
        if ball_test is not None:
            csv_data = {"id": -1, "positif": ball_test}|data
            export_data(csv_filename, thresh, csv_data)
    
    return return_info


# Trouve toutes les informations sur une zone de mouvement unique
#   Renvoie tous les cercles trouvés ayant un score plus grand que min_score
def analyse(raw, delta, x, y, height, width, min_score, infos=False, show=False):
    if height <= 0 or width <= 0:
        return []
    global min_side_size
    LUM_BAS = 30  # Minimum gardé par le filtre passe-haut
    thresh = cv2.threshold(delta, LUM_BAS, 255, cv2.THRESH_BINARY)[1]
    colorthresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    # Analyse de chaque contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cercles = []
    nbrFormes = len(contours)
    if contours and show:
        cv2.imshow("analysed zone", raw)
    for contour in contours:
        tx,ty,tw,th = cv2.boundingRect(contour) # Rectangle englobant du contour
        zone = thresh[ty:ty+th, tx:tx+tw]
        if min(tw, th) > min_side_size: # Si le contour est trop petit on l'ignore
            cercle = analyse_contour(colorthresh.copy(), contour, zone, tx, ty, tw, th, nbrFormes, infos)
            if cercle is not None and cercle.score > min_score:
                (cx, cy) = cercle.center
                cercle.center = (cx + x, cy + y) # On décale le centre par rapport à l'emplacement de la zone
                cercles.append(cercle)
    if contours and show:
        cv2.destroyWindow("analysed zone")
    return cercles


# Exporte une image de seuil "thresh" d'un contour
#   Exporte aussi les données "data" du contour dans le fichier csv "filename"
def export_data(filename, thresh, data):
    print(data)
    img_dir = 'data/img_contour'
    l = len(os.listdir(img_dir))
    data["id"] = l
    if os.path.isfile(filename): # Le fichier csv existe déjà
        csvfile = open(filename, "a", newline="")
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=list(data.keys()))
    else:  # Le fichier n'existe pas encore
        csvfile = open(filename, "w", newline="")
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=list(data.keys()))
        writer.writeheader()
    cv2.imwrite(f"{img_dir}/{l}.png", thresh) # Exporte l'image
    writer.writerow(data) # Exporte les données
    csvfile.close()
        

# Trouve les contours des zones de mouvement et actualise avg_frame
def get_contours(delta, avg_delta, gray, avg_frame, frames_seen):
    LOWPASS = 10     # Borne inférieure du passe-haut
    HIGHPASS = 255   # Borne supérieure du passe-haut
    dilateSize = 20  # Taille de dilatation des contours

    threshold_prec = cv2.threshold(delta, LOWPASS, HIGHPASS, cv2.THRESH_BINARY)[1]  # Passe-haut appliqué
    threshold_prec = cv2.dilate(threshold_prec,None,iterations = dilateSize)  # Formes dilatées (regroupe les formes)
    low_threshold = cv2.bitwise_not(cv2.dilate(threshold_prec,None,iterations = dilateSize))  # Parties de l'image qui n'ont pas bougé
    threshold_prec = cv2.erode(threshold_prec, None, iterations = dilateSize) # Formes recompressées

    threshold_avg = cv2.threshold(avg_delta, LOWPASS, HIGHPASS, cv2.THRESH_BINARY)[1] # Pareil mais avec l'image moyenne
    threshold_avg = cv2.dilate(threshold_avg,None,iterations = dilateSize)
    threshold_avg = cv2.erode(threshold_avg, None, iterations = dilateSize)

    threshold = cv2.bitwise_and(threshold_avg, threshold_prec)  # Correction de threshold avec l'image moyenne
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Mise à jour de l'image moyenne
    still_gray = cv2.bitwise_and(gray, low_threshold)  # Parties de l'image qui n'ont pas bougé, peint avec gray (infos à ajouter)
    missing = cv2.bitwise_and(avg_frame, cv2.bitwise_not(low_threshold))  # Parties de l'image qui ont bougé, peint avec avg_frame
    gray_no_movement = cv2.addWeighted(still_gray, 1, missing, 1, 0)  # Concaténation des 2
    alpha = 2.5/(frames_seen)  # Calcul de l'importance de l'information à ajouter (plus on a vu d'images, moins on a besoin de changer avg_frame)
    new_avg_frame = cv2.addWeighted(gray_no_movement, alpha, avg_frame, 1-alpha, 0.0)  # Mise à jour de avg_frame

    return (contours, new_avg_frame, threshold)



def main(filename, scaledown, frame_offset, INFOS_CONTOURS=False, INFOS_TRACKS=False, show=False):
    ##################################################################
    #### Définit les variables importantes et initialise la vidéo ####
    ##################################################################
    video = cv2.VideoCapture(f'data\\{filename}.mp4')
    global min_circle_score
    if video.isOpened():  
        vidwidth, vidheight = int(video.get(3)), int(video.get(4))
        vidwidthscale = int(vidwidth * scaledown)
        vidheightscale = int(vidheight * scaledown)
        vidfps = video.get(5)
        vidframe_count = int(video.get(7))
        print(vidwidthscale, vidheightscale)
        print(vidwidth, vidheight, vidfps, vidframe_count)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(f"data/output.mp4", fourcc, vidfps, (round(vidwidth),round(vidheight)))
        precedant, avg_frame, frames_seen, tracklist = None, None, 0, []

    # Skip les premières frames
    for i in range(frame_offset):
        check, frame = video.read()
        if not check:
            break

    while video.isOpened():
        ##############################################################
        #### Lit une image et ferme la vidéo si elle est terminée ####
        ##############################################################
        check, frame = video.read()
        if not check:
            break

        #####################################
        #### Preprocessing (gris + flou) ####
        #####################################
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # L'image en niveaux de gris
        gray = cv2.GaussianBlur(gray,(31,31), 0)  # Flou gaussien
        if frames_seen == 0:
            precedant, avg_frame = gray, gray
        frames_seen += 1
        print("Current frame:", frames_seen)

        delta = cv2.absdiff(precedant, gray) # Différence avec l'image précédente
        avg_delta = cv2.absdiff(avg_frame, gray) # Différence avec l'image moyenne
        precedant = gray # On met à jour precedant pour la prochaine image

        ####################################################
        #### Trouve les contours des zones de mouvement ####
        ####################################################
        contours, avg_frame, threshold = get_contours(delta, avg_delta, gray, avg_frame, frames_seen)

        ####################################################################
        #### Analyse les contours et relie les cercles aux trajectoires ####
        ####################################################################
        newtracks = []
        for contour in contours:
            ###########################################################
            #### Trouve tous les cercles dans la zone de mouvement ####
            ###########################################################
            x,y,width,height = cv2.boundingRect(contour)
            zone = frame[y:y+height, x:x+width]
            zonedelta = delta[y:y+height, x:x+width]
            circles = analyse(zone, zonedelta, x, y, height, width, min_circle_score, INFOS_CONTOURS)

            ###############################################
            #### Connecte les cercles aux trajectoires ####
            ###############################################
            for t in tracklist:  # On itère sur les tracks plutôt que sur les cercles pour tout skip si le 1e test échoue
                if t.connect_box(x,y,width,height):  # 1e check rapide (la zone est connectée)
                    if INFOS_TRACKS: # On affiche la zone (et ses cercles) <Violet> et la track <Bleue> connectée
                        nf = frame.copy()
                        t.draw_connect_box(nf,x,y,width,height,Color.purple,2)
                        t.draw(nf, Color.blue, 3)
                    for c in circles:
                        cs = t.connect_score(c)
                        if cs > 0:          # 2e check plus précis
                            new = t.copy()
                            new.add_circle(c,cs)
                            newtracks.append(new)
                            t.check = True
                            circles.remove(c)
                            draw_color = Color.green
                        else:
                            draw_color = Color.red
                        if INFOS_TRACKS:
                            c.draw(nf, draw_color, 2)
                            cv2.putText(nf, str(round(cs,2)), tuple(map(round, c.center)), cv2.FONT_HERSHEY_SIMPLEX, 2, draw_color, 3)
                    if INFOS_TRACKS:
                        cv2.imshow('Normal track connect', cv2.resize(nf, [vidwidthscale, vidheightscale]))
                        cv2.waitKey(0)
            
            #############################################################
            #### Nouvelle trajectoire pour chaque cercle non couvert ####
            #############################################################
            for c in circles:
                newtracks.append(Track(c))
        if INFOS_TRACKS:
            try:   # On fait un try except au cas où y'avait aucun contour
                cv2.destroyWindow('Normal track connect')
            except cv2.error:
                pass

        #######################################################
        #### Sauver les trajectoires probables mais ratées ####
        #######################################################
        for t in tracklist:
            if not t.check:
                idle_tracks = t.idle()
                newtracks += idle_tracks

        ##################################
        #### Dessine les trajectoires ####
        ##################################
        tracklist = newtracks
        outFrame = frame.copy()  # Frame qu'on va exporter dans la vidéo
        if tracklist:
            print(sorted(map(lambda t: t.score, tracklist)))
            bestTrajectory = max(tracklist, key = lambda t: t.score)
            print(bestTrajectory)
            for t in tracklist:
                # if t == bestTrajectory and t.score > 5:
                if t == bestTrajectory:
                    drawColor = Color.green
                    t.draw(outFrame, Color.green, -1, draw_score=False)  # Montre la meilleure trajectoire
                else:
                    drawColor = Color.red
                # t.draw(outFrame, drawColor, -1, draw_score=False)  # Montre toutes les trajectoires
        out.write(outFrame)

        if show:
            cv2.imshow('frame', cv2.resize(frame, [vidwidthscale, vidheightscale]))
            cv2.imshow('tracks', cv2.resize(outFrame, [vidwidthscale, vidheightscale]))

        if cv2.waitKey(0) & 0xFF == ord('q'):  # Quitter avec "q"
            break
    print("Exiting")
    out.release()


videos = ['videos/4_service', 'videos/4_top_smash', 'videos/4_paraboles', 'videos/4_block', 'videos/4_rien3', 'videos/1_smash_let', 'videos/4_rien2']
global kd_tree, kd_tree_criteria, kd_tree_normalization, csv_filename, nbr_voisins, min_side_size, min_circle_score
min_side_size = 30
min_circle_score = 0.4
csv_filename = "data/contours_tree.csv"
kd_tree_criteria = ["formes", "circularite", "taille", "circle", "solidite"]
kd_tree_normalization = [0.3, 20, 5, 0.001, 2]
nbr_voisins = 50
# kd_tree = classifieur.build_kd_tree("data/all.csv", kd_tree_criteria)
kd_tree = classifieur.build_kd_tree("data/contours_tree.csv", kd_tree_criteria)
main(videos[6], .29, frame_offset=0, INFOS_CONTOURS=False, INFOS_TRACKS=False, show=False)