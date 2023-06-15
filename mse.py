import cv2
import numpy as np

def similarity(path1, path2):
    try:
        # Load the two images
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
    
        # Compute the MSE between the two images
        mse = np.mean((img1 - img2) ** 2)

        # Compute the PSNR between the two images
        psnr = cv2.PSNR(img1, img2)

        print(f"The MSE between the two images is {mse}")
        print(f"The PSNR between the two images is {psnr}")

        return mse
    except:
        print(f'ERRO: {path1} ==> {path2}')