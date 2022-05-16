import random
import cv2

import albumentations as A
import glob

if __name__ == "__main__":
    im_number = 0
    for file_name in glob.glob('G:\Study\Dataset\clouds/train/*.png'):
        im_number += 1
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.augmentations.transforms.ChannelShuffle(0.5)
        ])
        image = cv2.imread(file_name)
        for i in range(5):
            augmented_image = transform(image=image)['image']
            cv2.imwrite(f'G:\Study\Dataset/clouds/augmented/{str(im_number) + str(i) + ".png"}', augmented_image)
