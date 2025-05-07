import cv2

# Lit une vidéo "filename" et permet de manuellement compter les images où le ballon est bien repéré
def lire(filename, scaledown, skip_frames):
    baseDirectory = "presentation/V2/Videos"
    video = cv2.VideoCapture(f'{baseDirectory}/{filename}.mp4')
    if video.isOpened():  
        vidwidth = int(video.get(3))
        vidwidthscale = int(vidwidth * scaledown)
        vidheight = int(video.get(4))
        vidheightscale = int(vidheight * scaledown)
        vidfps = video.get(5)
        vidframe_count = int(video.get(7))

    frame_infos = [0] * 9
    frame_count = 0

    for i in range(skip_frames):
        check, frame = video.read()
        frame_count += 1
        if not check:
            return

    while video.isOpened():
        check, frame = video.read()
        frame_count += 1
        if not check:
            break

        cv2.imshow('frame', cv2.resize(frame, [vidwidthscale, vidheightscale]))

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Quitter
            break
        elif key == ord('u'):  # Devant filet correct
            frame_infos[0] += 1
        elif key == ord('i'):  # Derrière filet correct
            frame_infos[1] += 1
        elif key == ord('o'):  # Pas en jeu correct
            frame_infos[2] += 1
        elif key == ord('j'):  # Devant filet pas repéré
            frame_infos[3] += 1
        elif key == ord('k'):  # Derrière filet pas repéré
            frame_infos[4] += 1
        elif key == ord('l'):  # Pas en jeu pas repéré
            frame_infos[5] += 1
        elif key == ord(','):  # Devant filet mal repéré
            frame_infos[6] += 1
        elif key == ord(';'):  # Derrière filet mal repéré
            frame_infos[7] += 1
        elif key == ord(':'):  # Pas en jeu mal repéré
            frame_infos[8] += 1

    print("Last frame:", "Résultats:", frame_infos)

videos = ["result 0 clean best track", "block", "rien 3", "rien 2"]
lire(videos[3], 0.20, 0)