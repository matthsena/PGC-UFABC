import cv2
import numpy as np

def similarity(path1, path2):
    try:
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # evitar erro de imagens de diferentes tamanhos
        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))

        fft1 = np.fft.fft2(img1)
        fft2 = np.fft.fft2(img2)

        spect1 = np.log(np.abs(fft1))
        spect2 = np.log(np.abs(fft2))

        spect_diff = cv2.absdiff(spect1, spect2)

        s = np.mean(spect_diff)

        return s

    except Exception as e:
        print(f'ERRO: {path1} ==> {path2} =>>> {str(e)}')