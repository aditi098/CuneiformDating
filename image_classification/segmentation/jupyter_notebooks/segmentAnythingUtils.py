import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
random.seed(41)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
def getFrontCutout(masks,image):
    frontMask = None
    if len(masks) == 0:
        return frontMask
    elif len(masks) == 1:
        frontMask = masks[0]
    elif len(masks) == 2:
        if masks[0]['bbox'][0] <=5 and masks[0]['bbox'][1] <=5:
            #masks[0] is background, return 1
            frontMask = masks[1]
        else:
            #choose between 0 or 1
            if masks[0]['area'] > masks[1]['area']*1.5: #masks[1] is much bigger than masks[2]
                frontMask = masks[0]
            elif masks[0]['bbox'][1] < masks[1]['bbox'][1]:  # ycoordinate of front mask will be smaller
                frontMask = masks[0]
            else:
                frontMask = masks[1]
    else:
        if masks[0]['bbox'][0] <=5 and masks[0]['bbox'][1] <=5:
            #masks[0] is background, choose from 1 or 2
            if masks[1]['area'] > masks[2]['area']*1.5: #masks[1] is much bigger than masks[2]
                frontMask = masks[1]
            elif masks[1]['bbox'][1] < masks[2]['bbox'][1]:  # ycoordinate of front mask will be smaller
                frontMask = masks[1]
            else:
                frontMask = masks[2]

        else:
            #choose between 0 or 1
            if masks[0]['area'] > masks[1]['area']*1.5: #masks[1] is much bigger than masks[2]
                frontMask = masks[0]
            elif masks[0]['bbox'][1] < masks[1]['bbox'][1]:  # ycoordinate of front mask will be smaller
                frontMask = masks[0]
            else:
                frontMask = masks[1]
                
    x,y,w,h = frontMask['bbox']
    x,y,w,h = int(x), int(y), int(w), int(h)
    cutout = image[y:y+h, x:x+w]
    return cutout

def resizeImage(image, max_dim):
    
    while image.shape[0]/max_dim > 1 or image.shape[1]/max_dim > 1:
        dim = (int(image.shape[1]/2), int(image.shape[0]/2))
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image 