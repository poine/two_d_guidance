#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, logging, numpy as np, matplotlib.pyplot as plt
import two_d_guidance as tdg

def make_real_path():
   _dir = '/home/poine/work/two_d_guidance/paths/demo_z'
   x0, x1 = -0.75, 2.25
   c1, r = [x1,0], 1.25
   line1 = tdg.make_line_path([x0, r], [x1, r], n_pt=100)
   circle1 = tdg.make_circle_path(c1, -r, np.pi/2, np.pi, n_pt=180)
   x2, y2 = 1.581, -0.35
   line2 = tdg.make_line_path([x1, -r], [x2, -r], n_pt=50)
   c2, r2, th2 = [x2, y2], 0.9, np.deg2rad(61.8)
   circle2 = tdg.make_circle_path(c2, r2, -np.pi/2, -th2, n_pt=90)
   c3, r3 = [0, -1.21], 0.9
   circle3 = tdg.make_circle_path(c3, r3, np.pi/2-th2, 2*th2, n_pt=90)
   c4, r4 = [-x2, y2], 0.9
   circle4 = tdg.make_circle_path(c4, r4, -(np.pi/2-th2), -th2, n_pt=90)
   line3 = tdg.make_line_path([-x1, -r], [-x2, -r], n_pt=50)
   circle5 = tdg.make_circle_path([-x1, 0], -r, -np.pi/2, np.pi, n_pt=180)
   line4 = tdg.make_line_path([-x1, r], [x0, r], n_pt=50)
   line1.append([circle1, line2, circle2, circle3, circle4, line3, circle5, line4])
   p = line1
   fname = os.path.join(_dir, 'track_trr_real.npz')
   p.save(fname)
   return fname, p



def make_small2_path():
   _dir = '/home/poine/work/two_d_guidance/paths/demo_z'


   x0, x1 = -0.75, 5
   c1, r = [x1,0], 1.75
   line1 = tdg.make_line_path([x0, r], [x1, r], n_pt=100)
   circle1 = tdg.make_circle_path(c1, -r, np.pi/2, np.pi, n_pt=180)
   r2 = 0.9575
   x2 = 4*r2
   line2 = tdg.make_line_path([x1, -r], [x2, -r], n_pt=50)
   circle2 = tdg.make_circle_path([x2, -(r-r2)], -r2, -np.pi/2, np.pi/2, n_pt=90)
   x3 = 2*r2
   circle3 = tdg.make_circle_path([x3, -(r-r2)], r2, 0, np.pi, n_pt=180)
   circle4 = tdg.make_circle_path([0, -(r-r2)], -r2, 0, np.pi, n_pt=180)
   circle5 = tdg.make_circle_path([-x3, -(r-r2)], r2, 0, np.pi, n_pt=180)
   circle6 = tdg.make_circle_path([-x2, -(r-r2)], -r2, 0, np.pi/2, n_pt=90)
   line3 = tdg.make_line_path([-x2, -r], [-x1, -r], n_pt=50)
   circle7 = tdg.make_circle_path([-x1, 0], -r, -np.pi/2, np.pi, n_pt=180)
   line4 = tdg.make_line_path([-x2, r], [x0, r], n_pt=50)
   line1.append([circle1, line2, circle2, circle3, circle4, circle5, circle6, line3, circle7, line4])

   p = line1
   fname = os.path.join(_dir, 'track_trr.npz')
   p.save(fname)
   return fname, p



    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    #fname, p = make_small2_path()
    fname, p = make_real_path()
    tdg.draw_path(plt.gcf(), plt.gca(), p)
    plt.show()

