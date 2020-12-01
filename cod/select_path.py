import sys
import numpy as np
import pdb
import cv2 as cv


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    path = [(line, col)]
    for i in range(E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = path[-1][1] + opt
        path.append((line, col))

    return path


def select_greedy_path(E):
    # pentru linia 0 alegem pixelul cu valoare minima
    line = 0
    col = np.argmin(E[line])
    path = [(line, col)]

    for i in range(1, E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        old_col = col

        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.argmin(E[line, old_col:old_col+2])
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.argmin(E[line, old_col-1:old_col+1]) - 1
        else:
            opt = np.argmin(E[line, old_col-1:old_col+2]) - 1

        # din opt scadem 1 pentru a trece de la intervalul (0,2) returnat de argmin la intervalul (-1,1)
        col = path[-1][1] + opt
        path.append((line, col))

    return path


def select_dynamic_programming_path(E):
    M = E.copy()
    # calculam dinamic matricea cu costul drumurilor
    for i in range(1, M.shape[0]):
        for j in range(M.shape[1]):
            # alegem minimul de pe coloana anterioara in functie de pozitia coloanei
            if j == 0:
                mini = min(M[i - 1, j:j + 2])
            elif j == M.shape[1] - 1:
                mini = min(M[i - 1, j - 1:j + 1])
            else:
                mini = min(M[i - 1, j - 1:j + 2])
            M[i][j] += mini

    # incepem de pe ultima linie si alegem pixelul cu valoare minima
    line = M.shape[0] - 1
    col = np.argmin(M[line])
    path = [(line, col)]

    for i in range(M.shape[0] - 2, -1, -1):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        old_col = col

        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.argmin(M[line, old_col:old_col + 2])
        elif path[-1][1] == M.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.argmin(M[line, old_col - 1:old_col + 1]) - 1
        else:
            opt = np.argmin(M[line, old_col - 1:old_col + 2]) - 1

        # din opt scadem 1 pentru a trece de la intervalul (0,2) returnat de argmin la intervalul (-1,1)
        col = path[-1][1] + opt
        path.append((line, col))

    # returnam drumul de la prima linie la ultima linie
    path.reverse()
    return path


def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_programming_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)