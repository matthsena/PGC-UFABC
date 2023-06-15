import lpips
import cv2

def similarity(path1, path2):
    # Load the two images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # evitar erro de imagens de diferentes tamanhos
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))

    # Create the LPIPS model
    loss_fn = lpips.LPIPS(net='alex')

    # Compute the LPIPS distance between the two images
    dist = loss_fn.forward(img1, img2).item()

    return dist