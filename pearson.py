import cv2
import numpy as np

def img_hist(img):
  hist = cv2.calcHist([img], [0], None, [256], [0, 256])

  return hist

def pearson_correlation(hist_a, hist_b):
  hist_a = cv2.normalize(hist_a, hist_a).flatten()
  hist_b = cv2.normalize(hist_b, hist_b).flatten()

  corr_coef = np.corrcoef(hist_a, hist_b)[0, 1]
  return corr_coef

def similarity(path1, path2):
    try: 
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        hist1 = img_hist(img1)
        hist2 = img_hist(img2)

        s = pearson_correlation(hist1, hist2)

        return s
    except:
        print(f'ERRO: {path1} ==> {path2}')
