#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, logging, numpy as np, matplotlib.pyplot as plt
import two_d_guidance as tdg

def run(which='line_01'):
   paths = {
      # ccw 1m circle
      'circle_02':  lambda: tdg.make_circle_path([2, 1.8], 1., 0, 2*np.pi, 360),
      # cw 1m circle
      'circle_03':  lambda: tdg.make_circle_path([2, 1.8], -1., 0, 2*np.pi, 360),
      # figure of eight
      'fig_of_eight_01': lambda: tdg.make_fig_of_height_path2(0.7),
      # oval 
      'oval_01': lambda: tdg.make_oval_path([-1.25, 0.], [1., 0.], 1.5),
      'oval_02': lambda: tdg.make_oval_path([-2.5, 0.], [2.5, 0.], 1.75),
      # line 1m 
      'line_01': lambda: tdg.make_line_path([-2, -1.8], [0., -0.8])

      
   }
   _dir = '/home/poine/work/two_d_guidance/paths/demo_z'
   p = paths[which]()
   p.transform([2, 1.8])
   fname = os.path.join(_dir, which+'.npz')
   p.save(fname)
   return fname, p

def convert_ethz():
    p = tdg.Path(load='/home/poine/work/oscar.git/oscar/oscar_control/paths/track_ethz_dual_01.npz')
    p.transform([1.8, 0])
    p.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/demo_z/track_ethz_cam1.npz')
    tdg.draw_path(plt.gcf(), plt.gca(), p)
    plt.show()

def convert_ethz2():
    p = tdg.Path(load='/home/poine/work/oscar.git/oscar/oscar_control/paths/demo_z/track_ethz_cam1_new_orig.npz')
    p.transform([-0.6, 0])
    p.save('/home/poine/work/oscar.git/oscar/oscar_control/paths/demo_z/track_ethz_cam1_new.npz')
    tdg.draw_path(plt.gcf(), plt.gca(), p)
    plt.show()
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300, suppress=True)
    if 1:
       fname, p = run('oval_02')
       tdg.draw_path(plt.gcf(), plt.gca(), p)
       plt.show()
    else:
       convert_ethz2()
