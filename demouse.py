import sys
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

sys.path.append("..")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# predictor = SamPredictor(sam)
image = cv2.imread('/Users/yanguosun/Developer/segment-anything/4a8cb432833d5b3c-2.jpg')
# predictor.set_image(image)
# masks, _, _ = predictor.predict(<input_prompts>)


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

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(masks)
print(len(masks))
print(masks[0].keys())
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 