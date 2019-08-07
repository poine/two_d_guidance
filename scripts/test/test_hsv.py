#!/usr/bin/env python
import numpy as np, cv2
import matplotlib.pyplot as plt
import timeit

import pdb

import two_d_guidance.trr_vision_utils as trr_vu


def make_patches(hsv_range, nh=4, size=50):
    pdb.set_trace()
    hmin, smin, vmin = hsv_range[0][0]
    hmax, smax, vmax = hsv_range[0][1]
    ds, dv = smax-smin, vmax-vmin
    #pdb.set_trace()
    hs = np.linspace(hmin, hmax, nh).astype(np.uint8)
    hsv_imgs = []
    for h in hs:
        hsv_imgs.append(np.zeros((size, size, 3), dtype=np.uint8))
        for x in range(size):
            for y in range(size):
                s = smin + int(dv*float(x)/size)
                v = vmin + int(ds*float(y)/size)
                hsv_imgs[-1][y,x] = h, s, v
    return hs, hsv_imgs, smin, ds, vmin, dv


def analyse_patches(hsv_imgs):
    print('# HSV analysis')
    for i, im in enumerate(hsv_imgs):
        print('patch {}'.format(i))
        for chan, _n in zip(range(3), ('h', 's', 'v')):
            _min, _max, _x, _y = cv2.minMaxLoc(im[:,:,chan])
            print('{}: {} {} ({},{})'.format(_n, _min, _max, _x, _y))
    
def plot_patches(hs, hsv_imgs):
    nc, nr = 2, 2
    fig, axs = plt.subplots(nc, nr, sharex=True, sharey=True)
    def x_format_func(value, tick_number): return smin + value/size*ds
    def y_format_func(value, tick_number): return vmin + value/size*dv
    for i, (hsv_img, ax) in enumerate(zip(hsv_imgs, axs.flat)):
        rgb_of_hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        ax.imshow(rgb_of_hsv_img)
        ax.set_title('h: {}'.format(hs[i]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(x_format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(y_format_func))
        ax.xaxis.set_label_text('s')
        ax.yaxis.set_label_text('v')

def main():
    if 0:
        w, h = 600, 300
        bgr_img = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            for y in range(h):
                bgr_img[y,x] = 0, 0, 255  
        for chan, _n in zip(range(3), ('b', 'g', 'r')):
            _min, _max, _x, _y = cv2.minMaxLoc(bgr_img[:,:,chan])
            print('{}: {} {} ({},{})'.format(_n, _min, _max, _x, _y))

    print(trr_vu.hsv_range(175, 20, smin=0, smax=0, vmin=0, vmax=0))
    return

    #hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hsv_range = trr_vu.hsv_red_ranges()
    #hsv_range = trr_vu.hsv_green_range()
    #hsv_range = trr_vu.hsv_blue_range()
    print hsv_range
    size = 50
    hs, hsv_imgs, smin, ds, vmin, dv = make_patches(hsv_range, nh=4, size=50)
  
    analyse_patches(hsv_imgs)
    plot_patches(hs, hsv_imgs)

    #plt.figure()
    #plt.imshow(bgr_img)
    #cv2.imshow('bgr', bgr_img)
    #cv2.imshow('hsv', bgr_of_hsv_img)
    #cv2.waitKey(0)

    plt.show()

if __name__ == '__main__':
    main()
