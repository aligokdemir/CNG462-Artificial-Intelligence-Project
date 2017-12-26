
import numpy as np
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box

# Preprocessing the Images

class ImagePreprocessing(ObservationWrapper):
    
    def __init__(self, env, height = 64, width = 64, grayscale = True, crop = lambda img: img):
        super(ImagePreprocessing, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        ncolors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [ncolors, height, width])

    def _observation(self, img):
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims = True)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.
        return img
