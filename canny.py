import cv2
# import numpy as np


def similarity(path1, path2):
    try: 
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # evitar erro de imagens de diferentes tamanhos
        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))

        bordas_img1 = cv2.Canny(img1, 100, 200)
        bordas_img2 = cv2.Canny(img2, 100, 200)


        # print(f'B1: {np.array(bordas_img1).shape} \n')

        diferenca = cv2.absdiff(bordas_img1, bordas_img2)

        s = (bordas_img1.size - cv2.countNonZero(diferenca)) / bordas_img1.size

        return s
    except:
        print(f'ERRO: {path1} ==> {path2}')
