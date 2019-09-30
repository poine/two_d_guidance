#!/usr/bin/env python3
import sys, time, numpy as np
import scipy.signal
#import rospy
import matplotlib.pyplot as plt

# find /mnt/mint17/home/poine -name '*.py' -exec grep -Hi ref {} \; | grep -v android | grep -v sage | grep -i sat

import two_d_guidance.utils as tdg_u
import two_d_guidance.plot_utils as tdg_pu

def plot_ref(_time, Xr, _sp=None):
    ax = plt.subplot(3,1,1)
    plt.plot(_time, Xr[:,0])
    if _sp is not None: plt.plot(_time, _sp)
    tdg_pu.decorate(ax, 'vel')
    ax = plt.subplot(3,1,2)
    plt.plot(_time, Xr[:,1])
    tdg_pu.decorate(ax, 'accel')
    ax = plt.subplot(3,1,3)
    plt.plot(_time, Xr[:,2])
    tdg_pu.decorate(ax, 'jerk')

def run_ref(_ref, _time, _sp):
    Xr = np.zeros((len(_time), 3))
    for i in range(1, len(_time)):
        dt = _time[i] - _time[i-1]
        Xr[i] = _ref.run(dt, _sp[i])
    return Xr
    
def work():
    _time = np.arange(0, 10, 0.01)

    _sats = [6., 50.]  # accel, jerk
    _ref1 = tdg_u.SecOrdLinRef(omega=6, xi=0.9, sats=None)
    _ref2 = tdg_u.SecOrdLinRef(omega=6, xi=0.9, sats=_sats)

    #_sp = 1.*np.ones(len(_time))
    _sp = 4.*scipy.signal.square(_time*np.pi/3)
    Xr1 = run_ref(_ref1, _time, _sp)
    Xr2 = run_ref(_ref2, _time, _sp)

    plot_ref(_time, Xr1)
    plot_ref(_time, Xr2, _sp)
    plt.show()
        
def main(args):
    work()
      
if __name__ == '__main__':
    main(sys.argv)
