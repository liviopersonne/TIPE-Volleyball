import csv
from random import randint
import matplotlib.pyplot as plt
import cv2

# Classe de donnée d'arbre kd
class Data:
    # Crée une donnée avec une ligne "row" d'un fichier csv, cette ligne a "nbr_dims" dimensions
    def __init__(self, row, arguments):
        if row is None:
            return
        s = row[0]
        l = s.split(";")
        self.id = int(l[0])
        if l[1] == "True":
            self.positive = True
        elif l[1] == "False":
            self.positive = False
        self.payload = [float(l[i]) for i in arguments]

    # Crée un point de donnée en entrant ses paramètres (on ne sait pas si c'est le ballon)
    def create_point(payload):
        pt = Data(None, 0)
        pt.payload = payload
        pt.id = -1
        pt.positive = False
        return pt

    # Montre un point de données
    def __repr__(self):
        if self.id == -1:
            return f"ManualData"
        elif self.positive:
            return f"PosData({self.id})"
        else:
            return f"NegData({self.id})"

##########################################
#### Intéractions avec le fichier csv ####
##########################################

# Itère sur les données du fichier "filename"  qui est un csv
def iterdata(filename, criteria):
    with open(filename, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)[0].split(";")  #Liste ordonnée des en-têtes
        arguments = []
        for crit in criteria:
            ind = headers.index(crit)
            arguments.append(ind)
        for row in reader:
            yield(Data(row, arguments))

# Liste des n premiers élements de "filename"
def first_data(filename, criteria, n):
    l = []
    data = iterdata(filename, criteria)
    for i in range(n):
        l.append(next(data))
    return l

# Switch 2 élements d'une liste
def switch(l, x, y):
    n = len(l)
    assert(x < n and y < n)
    save = l[x]
    l[x] = l[y]
    l[y] = save

# Vérifie si une liste est ordonnée
def ordered(l):
    for i in range(len(l) - 1):
        if l[i] > l[i+1]:
            return False
    return True

# Partitionne la data list l[s:f] autour de l'élément i sur la d-ième dimension
def partitionner(l, s, f, i, d):
    n = len(l)
    assert(ordered([0, s, i, f, n-1]) and 0 <= d)
    pivot = l[i]
    switch(l, s, i)
    a = s+1
    b = f
    while(a <= b):
        if l[a].payload[d] >= pivot.payload[d]:
            switch(l, a, b)
            b -= 1
        else:
            a += 1
    switch(l, s, b)
    return b

# Sélectionne une valeur de rang r dans la liste l[a:b] dans la dimension d
def selection_sous_liste(l, a, b, r, d):
    n = len(l)
    assert(ordered([0, a, b, n-1]) and 0 <= d and ordered([0, r, b-a]))
    if a < b:
        p = randint(a, b)
        q = partitionner(l, a, b, p, d)
        if q == r+a:
            return l[q]
        elif q < r+a:
            return selection_sous_liste(l, q+1, b, r-q+a-1, d)
        else:
            return selection_sous_liste(l, a, q-1, r, d)
    else:
        return l[a]

# Sélectionne une valeur de rang r dans la liste l dans la dimension d
def selection_liste(l, r, d):
    return selection_sous_liste(l, 0, len(l)-1, r, d)
            
# Affiche les composantes sur la dimension d des élements d'une liste de data: l
# d = -2 -> trie selon x.payload
# d = -1 -> affiche toutes les dimensions
def print_components(l, d = -1):
    dims = len(l[0].payload)
    assert(d < dims)
    if d >= 0:
        print("Dim:", d, end = " ")
    elif d == -2:
        for i in range(dims):
            print_components(sorted(l, key = lambda x: x.payload[i]), i)
    print("[", end = "")
    for i in l:
        if d == -1:
            print(i.id, "\b:", i.payload, end = ", ")
        else:
            print(i.id, "\b:", i.payload[d], end = ", ")
    print("\b\b]")

# Ajoute un critère dans le fichier csv
def add_criteria(filename, criteria_name, criteria_function):
    outfilename = filename.removesuffix(".csv") + "_new.csv"
    with open(filename, "r", newline="") as infile:
        with open(outfilename, "w", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            headers = next(reader)[0].split(";")
            writer.writerow([(";").join(headers + [criteria_name])])
            for row in reader:
                info = row[0].split(";")
                img_name = f"data/img/{info[0]}.png"
                img = cv2.imread(img_name, 1)
                crit_result = criteria_function(img)
                writer.writerow([(";").join(info + [crit_result])])



#######################################
#### Intéractions avec un arbre kd ####
#######################################

class kd_tree:
    def __init__(self, x, dim, g, d):
        self.x = x
        self.dim = dim
        self.g = g
        self.d = d

# Crée un arbre kd avec les données de l
def build_kd_tree_from_list(l, nbr_dims, dimension = 0):
    n = len(l)
    if n == 0:
        return None
    elif n == 1:
        return kd_tree(l[0], dimension, None, None)
    else:
        indice_mid = n//2
        mediane = selection_liste(l, indice_mid, dimension)
        liste_g = l[:indice_mid]
        liste_d = l[indice_mid+1:]
        next_dimension = (dimension + 1) % nbr_dims
        return kd_tree(mediane, dimension, build_kd_tree_from_list(liste_g, nbr_dims, next_dimension), build_kd_tree_from_list(liste_d, nbr_dims, next_dimension))

# Crée un arbre kd avec toutes les données du fichier "filename", si n!=-1, on ne prend que les n premières données
def build_kd_tree(filename, criteria, n = -1):
    if n == -1:
        return build_kd_tree_from_list(list(iterdata(filename, criteria)), len(criteria))
    else:
        return build_kd_tree_from_list(list(first_data(filename, criteria, n)), len(criteria))

# Distance entre 2 points de data (carré de la norme 2)
def dist(a, b, normalization):
    d = 0
    for i in range(len(a.payload)):
        d += abs(a.payload[i] - b.payload[i]) * normalization[i]
    return d

# Trouve les k plus proches voisins de x dans un arbre kd "t"
# Si "distances" est vrai, on renvoie le tableau des (point, distance)
# Si "distances" est faux, on renvoie le tableau des points
def k_ppv(t, x, k, normalization, distances = False):
    if t is None or k < 0:
        return []
    dimension = t.dim
    (t1, t2) = t.g is None, t.d is None
    if (t1, t2) == (True, True):
        if distances:
            return [(t.x, dist(x, t.x, normalization))]
        else:
            return [t.x]
    
    # Distance minimale entre x et un pt du "mauvais" côté de l'arbre kd
    # C'est la distance entre x et son projeté orthogonal sur l'axe de séparation
    distmin = abs(t.x.payload[t.dim] - x.payload[t.dim]) * normalization[t.dim]

    if x.payload[t.dim] < t.x.payload[t.dim]:
        good = t.g  # Arbre du "bon côté"
        bad = t.d
    else:
        good = t.d
        bad = t.g
    
    points_candidats = k_ppv(good, x, k, normalization, True)
    points_candidats.append((t.x, dist(x, t.x, normalization)))
    points_surs = []
    for (point, distance) in points_candidats:
        if distance <= distmin:   # Aucun point du "mauvais côté" ne sera plus proche
            points_surs.append((point, distance))
            points_candidats.remove((point, distance))

    reste = k - len(points_surs) # Nombre de points à rajouter au résultat final
    points_candidats += k_ppv(bad, x, reste, normalization, True) # On regarde de l'autre côté de la séparation
    points_candidats.sort(key = lambda x: x[1]) # On trie les candidats par distance croissance
    points_surs += points_candidats[:reste] # On prend ce qu'il reste

    if distances:
        return points_surs
    else:
        return list(map(lambda x: x[0], points_surs))

# Score d'une donnée "x" dans l'arbre kd "t": proportion de points positifs parmi les "k" plus proches voisins
def score_ppv(t, x, k, normalization):
    assert(k > 0)
    ppv = k_ppv(t, x, k, distances=False, normalization=normalization)
    list_pos = list(filter(lambda x: x.positive, ppv))
    return len(list_pos) / len(ppv)
    

###############################
#### Fonctions d'affichage ####
###############################

# Affiche un arbre kd
def print_kd_tree(t, prefixe = ""):
    if t is None:
        return
    else:
        print(t.dim, t.x)
        if t.g is not None:
            print(prefixe, "g:", end = " ")
            print_kd_tree(t.g, prefixe + "  ")
        if t.d is not None:
            print(prefixe, "d:", end = " ")
            print_kd_tree(t.d, prefixe + "  ")

def aux_3d_tree(t, ax, dims, normalization = [1, 1, 1]):
    p = t.x.payload
    (x, y, z) = (p[dims[0]]*normalization[0], p[dims[1]]*normalization[1], p[dims[2]]*normalization[2])
    if t.x.positive:
        marker = "."
        color = "g"
    else:
        marker = "."
        color = "r"
    ax.scatter(x, y, z, marker=marker, color=color)
    if t.g is not None:
        aux_3d_tree(t.g, ax, dims, normalization)
    if t.d is not None:
        aux_3d_tree(t.d, ax, dims, normalization)

# Affiche un arbre kd en 3d
# Params: t -> arbre_kd || dims -> indice des 3 critères affichés || dimsName -> nom des critères affichés (dans le même ordre)
def show_3d_kd_tree(t, dims, criteria = None, normalization = [1, 1, 1], lims = None):
    assert(len(dims) == 3 and len(normalization) == 3)
    fig = plt.figure()
    if lims is not None and len(lims) == 3:
        ax = fig.add_subplot(projection='3d', autoscale_on=False, xlim=lims[0], ylim=lims[1], zlim=lims[2])
    else:
        ax = fig.add_subplot(projection='3d')
        
    aux_3d_tree(t, ax, dims, normalization)

    if criteria is not None:
        ax.set_xlabel(criteria[dims[0]])
        ax.set_ylabel(criteria[dims[1]])
        ax.set_zlabel(criteria[dims[2]])
        
    plt.show()

# Affiche les n premiers points sur un axe selon criteria (vert = balle / rouge = pas balle)
def show_1d_points(filename, criteria, n, lim = None):
    points = first_data(filename, [criteria], n)
    fig = plt.figure()
    if lim is not None:
        ax = fig.add_subplot(xlim = lim, ylim = (-1, 100))
    else:
        ax = fig.add_subplot(ylim = (-1, 100))
    ax.set_xlabel(criteria)
    for p in points:
        if p.positive:
            ax.scatter(p.payload[0], 4, marker=".", color="g")
        else:
            ax.scatter(p.payload[0], 0, marker=".", color="r") 
    plt.show()






if __name__ == '__main__':
    filename = "data/contours_tree.csv"
    

    # l = first_data(filename, ["solidite", "white", "frame"], 10)

    criteria = ["formes", "circularite", "taille", "circle", "solidite"]
    t = build_kd_tree(filename, criteria, 200)
    # print_kd_tree(t)
    ## show_3d_kd_tree(t, [0, 1, 3], criteria, [0.3, 20, 0.001], [(0, 15), (0, 20), (0, 10)])
    show_3d_kd_tree(t, [0, 1, 2], criteria, [0.3, 20, 5], [(0, 15), (0, 20), (4, 10)])
    # show_3d_kd_tree(t, [0, 2, 3], criteria, [0.3, 5, 0.001], [(0, 15), (4, 10), (0, 10)])
    # show_3d_kd_tree(t, [1, 2, 4], criteria, [20, 5, 2], [(0, 20), (4, 10), (0, 2)])

    # show_1d_points(filename, "area", 500, (0, 5000))


    def criteria(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
        components, _ = cv2.connectedComponents(img)
        # print(components)
        # cv2.imshow("image", img)
        # cv2.waitKey(0) & 0xFF
        # cv2.destroyWindow("image")
        return str(components)
    # add_criteria(filename, "connexe", criteria)


    # normalization = [0.3, 20, 15]
    # l = first_data(filename, ["formes", "circularite", "taille"], 100)
    # t = build_kd_tree(filename, ["formes", "circularite", "taille"], 100)
    # k = 10
    # test_point = Data.create_point([7, 0.14885844539616933, 1.3989481253464908])
    # print(sorted([(p, dist(p, test_point, normalization)) for p in l], key = lambda x: x[1])[:k])
    # print(k_ppv(t, test_point, k, normalization, True))
    # print(score_ppv(t, test_point, k, normalization))





