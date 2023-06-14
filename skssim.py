# similarity between two images with scikit-image
import cv2
from skimage.metrics import structural_similarity as ssim


def similarity(path1, path2):
    try:
        # Load the two images
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        # evitar erro de imagens de diferentes tamanhos
        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))

        # Convert the images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Compute the SSIM between the two images
        similarity = ssim(gray1, gray2)

        return similarity
    except:
        print(f'ERRO: {path1} ==> {path2}')