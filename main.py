from PIL import Image, ImageOps
from scipy.fft import fft2, ifft2
import numpy as np

def read_image(filename):
    im1 = Image.open(filename)
    im2 = ImageOps.grayscale(im1)
    return im2

def image_fft(img):
    temp_array = np.array(img)
    temp_array = fft2(temp_array)
    return temp_array

def fft_shift(frequency_matrix):
    return np.fft.fftshift(frequency_matrix)

def image_ifft(frequency_matrix):
    unshifted = np.fft.ifftshift(frequency_matrix)
    return ifft2(unshifted)

if __name__ == '__main__':
    img1 = read_image('building.bmp')
    fft1 = image_fft(img1)
    fft1_s = fft_shift(fft1)
    print(type(fft1))
    print(fft1.shape)
    im_out = Image.fromarray(np.uint8(abs(fft1)))
    im_out.show()
    print(type(fft1_s))
    print(fft1_s.shape)
    im_out2 = Image.fromarray(np.uint8(abs(fft1_s)))
    im_out2.show()
