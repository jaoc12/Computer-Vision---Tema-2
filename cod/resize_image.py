import sys
import cv2 as cv
import numpy as np
import copy

from parameters import *
from select_path import *

import pdb


def compute_energy(img):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param img: imaginea initiala
    :return:E - energia
    """
    # urmati urmatorii pasi:
    # 1. transformati imagine in grayscale
    # 2. folositi filtru sobel pentru a calcula gradientul in directia X si Y
    # 3. calculati magnitudinea pentru fiecare pixel al imaginii

    E = np.zeros((img.shape[0], img.shape[1]))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1)
    E = np.abs(sobelx) + np.abs(sobely)
    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    E = compute_energy(img)
    new_image_E = img.copy()
    new_image_E[:, :, 0] = E.copy()
    new_image_E[:, :, 1] = E.copy()
    new_image_E[:, :, 2] = E.copy()

    for row, col in path:
        new_image_E[row, col] = color
    cv.imshow('path img', np.uint8(new_image))
    cv.imshow('path E', np.uint8(new_image_E))
    cv.waitKey(1000)


def delete_path(img, path):
    """
    elimina drumul vertical din imagine
    :param img: imaginea initiala
    :path - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col = path[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        updated_img[i, col:] = img[i, col+1:].copy()
        
    return updated_img


def decrease_width(params: Parameters, num_pixels, r=None):
    # copiaza imaginea originala
    img = params.image.copy()
    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i+1, num_pixels))

        E = compute_energy(img)
        # fortam drumul sa treaca prin zona marcata pentru stergere
        # scazand o valoare mare din ea
        if params.resize_option == "eliminaObiect":
            x, y, w, h = r
            # fereastra incepe in coltul (x,y), are mereu h linii
            # insa numarul de coloane scade cu 1 pentru fiecare pas al programului
            E[x: x+h, y: y+w-i] += -1000
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img


def decrease_height(params: Parameters, num_pixels):
    h, w, c = params.image.shape

    # rotim imaginea
    img = np.array([[params.image[j][i] for j in range(h-1, -1, -1)] for i in range(w)])
    # copiem imaginea originala
    img_original = params.image.copy()

    # apelam micsorarea pe latime pentru imaginea rotita
    params.image = img
    img = decrease_width(params, num_pixels)

    cv.destroyAllWindows()
    h, w, c = img.shape
    # rotim inapoi imaginea micsorata
    img = np.array([[img[j][i] for j in range(h)] for i in range(w-1, -1, -1)])
    # punem imaginea originala inapoi in params
    params.image = img_original.copy()
    return img


def image_amplification(params: Parameters):
    # calculam noile dimensiuni si cream imaginea redimensionata
    h_old, w_old, c = params.image.shape
    h_new = int(params.factor_amplification * h_old)
    w_new = int(params.factor_amplification * w_old)
    image_resized = cv.resize(params.image, (w_new, h_new))

    # copiem imaginea originala
    original_image = params.image.copy()

    # readucem imaginea la latimea initiala
    params.image = image_resized
    image_resized_width = decrease_width(params, w_new-w_old)

    # readucem imaginea la inaltimea initiala
    params.image = image_resized_width
    image_resized_width_height = decrease_height(params, h_new-h_old)

    # returnam imaginea amplificata, impreuna cu imaginea originala din params
    params.image = original_image.copy()
    return image_resized_width_height


def delete_object(params: Parameters, x, y, w, h):
    original_image = params.image.copy()
    image_resized = decrease_width(params, w, r=(x, y, w, h))
    params.image = original_image.copy()
    return image_resized


def resize_image(params: Parameters):

    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image

    elif params.resize_option == 'micsoreazaInaltime':
        resized_image = decrease_height(params, params.num_pixel_height)
        return resized_image
    
    elif params.resize_option == 'amplificaContinut':
        resized_image = image_amplification(params)
        return resized_image

    elif params.resize_option == 'eliminaObiect':
        # obtinem coordonatele si dimensiunea ferestrei de eliminat
        y, x, w, h = cv.selectROI(np.uint8(params.image))
        resized_image = delete_object(params, x, y, w, h)
        return resized_image

    else:
        print('The option is not valid!')
        sys.exit(-1)