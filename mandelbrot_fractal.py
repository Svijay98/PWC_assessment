import numpy as np 
from PIL import Image
import cv2

import colorsys


#  VERSION 1
'''
#Setting width of image output
WIDTH = 1024


#Function: return a tuple of colors
# as integer value of rgb

def rgb_conv(i):
    color = 255* np.array(colorsys.hsv_to_rgb(i/255.0, 1.0, 0.5))
    return tuple(color.astype(int))

def mandelbrot(x,y):
    c0 = np.complex(x,y)
    c =0
    for i in range(1000):
        if abs(c) >2:
            return rgb_conv(i)
        c = c*c +c0
    return (0,0,0)

img = Image.new('RGB', (WIDTH, int(WIDTH / 2)))
pixels = img.load()

for x in range(img.size[0]):

    print("%.2f %%" %(x/WIDTH * 100.0))
    for y in range(img.size[1]):
        pixels[x,y] = mandelbrot((x - (0.75 *WIDTH))/ (WIDTH / 4), 
                                        (y -(WIDTH/4))/(WIDTH/4))
img.show()
'''


#  VERSION 2
# drawing area
xa = -2.0
xb = 1.0
ya = -1.5
yb = 1.5

#max iteratrions
maxIt = 255

#image size

imgx = 512
imgy = 512
image = Image.new('RGB', (imgx, imgy))

for y in range(imgy):
    zy = y *(yb-ya)/ (imgy -1) +ya
    for x in range(imgx):
        zx = x*(xb-xa) /(imgx -1) +xa
        z = zx +zy*1j
        c = z
        for i in range(maxIt):
            if abs(z) >2.0: break
            z = z*z +c
        image.putpixel((x,y), (i %4*64, i%8*32, i%16*16))

image.show()
image.save("mandelbrot2.png")

