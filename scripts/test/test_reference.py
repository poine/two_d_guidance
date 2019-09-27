#!/usr/bin/env python
import sys, time, numpy as np
import rospy
import matplotlib.pyplot as plt

# find /mnt/mint17/home/poine -name '*.py' -exec grep -Hi ref {} \; | grep -v android | grep -v sage | grep -i sat

import two_d_guidance.utils as tdg_u
import two_d_guidance.plot_utils as tdg_pu

def work():
    _time = np.arange(0, 10, 0.01)

    _sats = [4., 25.]  # accel, jerk
    _ref = tdg_u.SecOrdLinRef(omega=6, xi=0.9, sats=_sats)

    _sp = 5*np.ones(len(_time))
    Xr = np.zeros((len(_time), 3))
    for i in range(1, len(_time)):
        dt = _time[i] - _time[i-1]
        Xr[i] = _ref.run(dt, _sp[i])

    ax = plt.subplot(3,1,1)
    plt.plot(_time, Xr[:,0])
    tdg_pu.decorate(ax, 'vel')
    ax = plt.subplot(3,1,2)
    plt.plot(_time, Xr[:,1])
    tdg_pu.decorate(ax, 'accel')
    ax = plt.subplot(3,1,3)
    plt.plot(_time, Xr[:,2])
    tdg_pu.decorate(ax, 'jerk')
    plt.show()
        
def main(args):
    work()
      
if __name__ == '__main__':
    main(sys.argv)
