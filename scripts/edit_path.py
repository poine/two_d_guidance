#!/usr/bin/env python3
import sys, os, time, math, numpy as np, matplotlib.pyplot as plt
import rospy, rospkg


'''
CLI interface for path manipulations
'''
import two_d_guidance as tdg



if __name__ == '__main__':
    tdg_wd = rospkg.RosPack().get_path('two_d_guidance')
    #_if = os.path.join(tdg_wd, 'paths/demo_z/track_ethz_cam1_new.npz')
    _if = os.path.join(tdg_wd, 'paths/roboteck/track.npz')
    _p = tdg.Path(load=_if)
    _pr = tdg.path_factory.make_reversed_path(_p)
    #_of = os.path.join(tdg_wd, 'paths/demo_z/track_ethz_cam1_ccw.npz')
    _of = os.path.join(tdg_wd, 'paths/roboteck/track_rev.npz')
    _pr.save(_of)
    tdg.draw_path(plt.gcf(), plt.gca(), _pr)
    plt.show()
    
