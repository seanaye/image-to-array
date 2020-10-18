from typing import Tuple
import skimage.io as skio
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.filters import threshold_mean
from skimage.color import rgb2gray
from numpy import ndarray


def get_file(path: str, size: Tuple[int, int]) -> ndarray:
    faces = skio.imread(path)
    return resize(faces, size, anti_aliasing=False)


def img_to_binary(img: ndarray, show: bool) -> ndarray:
    grayscale = rgb2gray(img)
    thresh = threshold_mean(grayscale)
    binary = grayscale > thresh
    if show:
        plt.gray()
        plt.imshow(binary)
        plt.show()
    return binary


if __name__ == '__main__':
    img = get_file('test.jpg', (32, 32))
    binary = img_to_binary(img, True)
    print(binary.tolist())

