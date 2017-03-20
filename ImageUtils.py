import numpy as np
import skimage
import skimage.transform
from utils import floatX
import matplotlib.pyplot as plt


class ImageHelper():
    def __init__(self, IMAGE_W):
        self.IMAGE_W = IMAGE_W
        self.MEAN_VALUES = np.array([104, 117, 123]).reshape((3, 1, 1))

    def prep_photo_and_art(self, photo_path, art_path):
        photo = plt.imread(photo_path)
        rawimphoto, photo = self.prep_image(photo)
        # plt.imshow(rawimphoto)

        art = plt.imread(art_path)
        rawimart, art = self.prep_image(art)
        # plt.imshow(rawimart)
        return photo, art



    def prep_image(self,im):
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.repeat(im, 3, axis=2)
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (self.IMAGE_W, w * self.IMAGE_W / h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h * self.IMAGE_W / w, self.IMAGE_W), preserve_range=True)

        # Central crop
        h, w, _ = im.shape
        im = im[h // 2 - self.IMAGE_W // 2:h // 2 + self.IMAGE_W // 2, w // 2 - self.IMAGE_W // 2:w // 2 + self.IMAGE_W // 2]

        rawim = np.copy(im).astype('uint8')

        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Convert RGB to BGR
        im = im[::-1, :, :]

        im = im - self.MEAN_VALUES
        return rawim, floatX(im[np.newaxis])


    def deprocess(self, x):
        x = np.copy(x[0])
        x += self.MEAN_VALUES

        x = x[::-1]
        x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

        x = np.clip(x, 0, 255).astype('uint8')
        return x