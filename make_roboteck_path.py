#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, logging, numpy as np, matplotlib.pyplot as plt
import two_d_guidance as tdg
import pdb

def make_square(res=0.01):
   _dir = tdg.tdg_dir()+'/paths/roboteck'
   dx, dy, r, m = 1.040, 0.64, 0.16, 0.04
   x10, y10, x11, y11 = m, r+m, m, dy-r-m
   line1 = tdg.make_line_path([x10, y10], [x11, y11], res=res)
   c1 = [r+m, dy-r-m]
   circle1 = tdg.make_circle_path(c1, -r, -np.pi, np.pi/2, res=res)
   x20, y20, x21, y21 = r+m, dy-m, dx-r-m, dy-m
   line2 = tdg.make_line_path([x20, y20], [x21, y21], res=res)
   c2 = [dx-r-m, dy-r-m]
   circle2 = tdg.make_circle_path(c2, -r, np.pi/2, np.pi/2, res=res)
   x30, y30, x31, y31 = dx-m, dy-r-m, dx-m, r+m
   line3 = tdg.make_line_path([x30, y30], [x31, y31], res=res)
   c3 = [dx-r-m, r+m]
   circle3 = tdg.make_circle_path(c3, -r, 0, np.pi/2, res=res)
   x40, y40, x41, y41 = dx-r-m, m, r+m, m
   line4 = tdg.make_line_path([x40, y40], [x41, y41], res=res)
   c4 = [r+m, r+m]
   circle4 = tdg.make_circle_path(c4, -r, -np.pi/2, np.pi/2, res=res)
   line1.append([circle1, line2, circle2, line3, circle3, line4, circle4])
   p = line1
   fname = os.path.join(_dir, 'square_1.npz')
   p.save(fname)
   return fname, p

def make_oval(res=0.01):
   _dir = tdg.tdg_dir()+'/paths/roboteck'
   r, m = 0.25, 0.125 # radius, margin
   dx, dy = [1.1, 0.7]
   xc, yc = [dx/2, dy/2]
   dxl = dx-2*(r+m)
   x10, y10, x11, y11 = xc-dxl/2, yc-r, xc+dxl/2, yc-r
   line1 = tdg.make_line_path([x10, y10], [x11, y11], res=res)
   c1 = [xc+dxl/2, yc]
   circle1 = tdg.make_circle_path(c1, r, -np.pi/2, np.pi, res=res)
   x20, y20, x21, y21 = xc+dxl/2, yc+r, xc-dxl/2, yc+r
   line2 = tdg.make_line_path([x20, y20], [x21, y21], res=res)
   c2 = [xc-dxl/2, yc]
   circle2 = tdg.make_circle_path(c2, r, np.pi/2, np.pi, res=res)
   line1.append([circle1, line2, circle2])
   p = line1
   fname = os.path.join(_dir, 'oval_1.npz')
   p.save(fname)
   return fname, p

def make_path(res=0.01):
   _dir = '/home/poine/work/two_d_guidance/paths/roboteck'
   r = 0.16
   x10, y10, x11, y11 = 0., 0., 0., 0.48  # moved start of path to center
   line1 = tdg.make_line_path([x10, y10], [x11, y11], res=res)
   c1 = [0.16, 0.48]
   circle1 = tdg.make_circle_path(c1, -r, -np.pi, np.pi, res=res)
   x2, y2, x3, y3 = 0.32, 0.48, 0.32, 0.32
   line2 = tdg.make_line_path([x2, y2], [x3, y3], res=res)
   c2 = [0.48, 0.32]
   circle2 = tdg.make_circle_path(c2, r, -np.pi, np.pi, res=res)
   x4, y4, x5, y5 = 0.64, 0.32, 0.64, 0.48
   line3 = tdg.make_line_path([x4, y4], [x5, y5], res=res)
   c3 = [0.8, 0.48]
   circle3 = tdg.make_circle_path(c3, -r, -np.pi, np.pi/2, res=res)
   x6, y6, x7, y7 = 0.8, 0.64, 0.88, 0.64
   line4 = tdg.make_line_path([x6, y6], [x7, y7], res=res)
   c4 = [0.88, 0.48]
   circle4 = tdg.make_circle_path(c4, -r, np.pi/2, np.pi/2, res=res)
   x8, y8, x9, y9 = 1.04, 0.48, 1.04, 0.16
   line5 = tdg.make_line_path([x8, y8], [x9, y9], res=res)
   c5 = [0.88, 0.16]
   circle5 = tdg.make_circle_path(c5, -r, 0, np.pi/2, res=res)
   x60, y60, x61, y61 = 0.88, 0., 0.32, 0.
   line6 = tdg.make_line_path([x60, y60], [x61, y61], res=res)
   c6 = [0.32, 0.16]
   circle6 = tdg.make_circle_path(c6, -r, -np.pi/2, np.pi/2, res=res)
   x70, y70, x71, y71 = 0.16, 0.16, 0.16, 0.48
   line7 = tdg.make_line_path([x70, y70], [x71, y71], res=res)
   line1.append([circle1, line2, circle2, line3, circle3, line4, circle4, line5, circle5, line6, circle6, line7])
   p = line1
   fname = os.path.join(_dir, 'track.npz')
   p.save(fname)
   return fname, p

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    #fname, p = make_path()
    #fname, p = make_oval()
    fname, p = make_square()
    tdg.draw_path(plt.gcf(), plt.gca(), p)
    plt.show()

