import lpips
import cv2
import torch

def similarity(path1, path2):
    try:
        # Load the two images
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        # evitar erro de imagens de diferentes tamanhos
        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))

        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2.0 - 1.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2.0 - 1.0

        # Create the LPIPS model
        # loss_fn = lpips.LPIPS(net='alex')
        loss_fn_vgg = lpips.LPIPS(net='vgg')

        # Compute the LPIPS distance between the two images
        dist = loss_fn_vgg.forward(img1, img2)

        return dist.item()
    except:
        print(f'ERROR: {path1} => {path2}')
        return 0