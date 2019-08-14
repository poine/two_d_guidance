#!/usr/bin/env python

import os, sys
import math, numpy as np
import rospy, rospkg

import two_d_guidance as tdg


#
# This is a specialized version of a path including start and finish landmarks
#
class StateEstPath(tdg.path.Path):
    def __init__(self, path_filename, xstart=1.21, xfinish=0.86):
        tdg.path.Path.__init__(self, load=path_filename)
        self.len = self.dists[-1] - self.dists[0]
        # start is 1.25m before center of straight line
        self.lm_start_idx, self.lm_start_point = self.find_point_at_dist_from_idx(0, _d=self.len-xstart)
        self.lm_start_s = self.dists[self.lm_start_idx]
        # finish is 1.25m after center of straight line
        self.lm_finish_idx, self.lm_finish_point = self.find_point_at_dist_from_idx(0, _d=xfinish)
        self.lm_finish_s = self.dists[self.lm_finish_idx]

    def report(self):
        rospy.loginfo(' path len {}'.format(self.len))
        rospy.loginfo(' path start {} (dist {}) '.format(self.points[0], self.dists[0]))
        rospy.loginfo(' path finish {}(dist {})'.format(self.points[-1], self.dists[-1]))
        rospy.loginfo('  lm_start idx {} pos {} dist {}'.format(self.lm_start_idx, self.lm_start_point, self.lm_start_s))
        rospy.loginfo('  lm_finish idx {} pos {} dist {}'.format(self.lm_finish_idx, self.lm_finish_point, self.lm_finish_s))

#
# Landmark crossing with hysteresis
#
class TrackMark:
    def __init__(self, s, name, hist=5):
        ''' landmark's abscice, name, and  '''
        self.s, self.name = s, name
        self.crossed = False
        self.cnt, self.hist = 0, hist
        
    def update(self, s, delta=0.1): # FIXME....
        if abs(s-self.s) < delta:
            if self.cnt < self.hist: self.cnt+=1 
            if self.cnt == self.hist and not self.crossed:
                self.crossed = True
                print('passing lm {}: prev {:.3f} lm {:.3f}'.format(self.name, self.s, s))
                return True
        else:
            if self.cnt > 0: self.cnt -=1
            if self.cnt == 0:
                self.crossed = False 
        
        return False
        
#
# Naive State estimation:
#    - integrate linear vel to predict abscice on path
#    - use landmarks (start and finish lines for now) as measurements
#
class StateEstimator:
    LM_START, LM_FINISH, LM_NB = range(3)
    lm_names = ['start', 'finish']
    
    def __init__(self, lm_passed_cbk=None):
        self.lm_passed_cbk = lm_passed_cbk
        self.load_path()
        self.s, self.sn = 0, 0.
        self.meas_dist_to_start, self.meas_dist_to_finish = float('inf'), float('inf')
        self.predicted_dist_to_start, self.predicted_dist_to_finish = float('inf'), float('inf')
        self.start_residual, self.finish_residual = float('inf'), float('inf')
        self.start_track = TrackMark(self.path.lm_start_s, 'start')
        self.finish_track = TrackMark(self.path.lm_finish_s, 'finish')
        self.last_stamp = None

    def load_path(self):
        tdg_dir = rospkg.RosPack().get_path('two_d_guidance')
        default_path_filename = os.path.join(tdg_dir, 'paths/demo_z/track_trr_real.npz')
        path_filename = rospy.get_param('~path_filename', default_path_filename)
        rospy.loginfo(' loading path: {}'.format(path_filename))
        self.path = StateEstPath(path_filename)
        self.path.report()
        
    def _update_s(self, ds):
        self.s += ds
        self.prev_sn = self.sn
        self._norm_s()
        
        if self.start_track.update(self.sn) and self.lm_passed_cbk is not None: self.lm_passed_cbk(StateEstimator.LM_START)

        if self.finish_track.update(self.sn) and self.lm_passed_cbk is not None: self.lm_passed_cbk(StateEstimator.LM_FINISH)
            
        
    def update_odom(self, seq, stamp, vx, vy, k=1.05):
        if self.last_stamp is not None:
            dt = (stamp - self.last_stamp).to_sec()
            #print('odom dt {} vx {:.4f} vy {:.4f}'.format(dt, vx, vy))
            if dt < 0.01 or dt > 0.1:
                print('state est: out of range dt')
            else:
                ds = k*vx*dt
                #print('update: {}'.format(ds))
                self._update_s(ds)
        self.last_stamp = stamp

    def _norm_s(self):
        self.sn = self.s
        while self.sn > self.path.len: self.sn -= self.path.len
        while self.sn < 0: self.sn += self.path.len

    def _norm_s_err(self, s_err):
        while s_err > self.path.len/2: s_err -= self.path.len
        while s_err < -self.path.len/2: s_err += self.path.len
        return s_err
        
        
    def update_landmark(self, meas_dist_to_start, meas_dist_to_finish, gain=0.075, disable_correction=False):
        #print('meas start {} meas finish {}'.format(meas_dist_to_start, meas_dist_to_finish))
        self.meas_dist_to_start, self.meas_dist_to_finish = meas_dist_to_start, meas_dist_to_finish
        if not math.isinf(meas_dist_to_finish) or disable_correction:
            self.predicted_dist_to_finish = self.path.lm_finish_s-self.sn
            self.finish_residual = self._norm_s_err(meas_dist_to_finish - self.predicted_dist_to_finish)
            self.finish_residual = np.clip(self.finish_residual, -0.5, 0.5)
            ds = -gain*self.finish_residual
            self._update_s(ds)
        else:
            self.predicted_dist_to_finish = float('inf')
            self.finish_residual = float('inf')
            
        if not math.isinf(meas_dist_to_start) or disable_correction:
            self.predicted_dist_to_start = self.path.lm_start_s-self.sn
            self.start_residual = self._norm_s_err(meas_dist_to_start - self.predicted_dist_to_start)
            self.start_residual = np.clip(self.start_residual, -0.5, 0.5)
            ds = -gain*self.start_residual
            self._update_s(ds)
        else:
            self.predicted_dist_to_start = float('inf')
            self.start_residual = float('inf')
        
        
    def status(self):
        txt  = 'state_est: sn {:.3f} s {:.3f}\n'.format(self.sn, self.s)
        #try:
        txt += '  start s: {:.2f} finish s: {:.2f}\n'.format(self.path.lm_start_s, self.path.lm_finish_s)
        txt += '  dist_to_finish meas {:.2f} pred {:.2f} res {}\n'.format(self.meas_dist_to_finish, self.predicted_dist_to_finish, self.finish_residual)
        txt += '  dist_to_start meas {:.2f} pred {:.2f} res {}\n'.format(self.meas_dist_to_start, self.predicted_dist_to_start, self.start_residual)
        #except AttributeError: pass
        return txt


    def dist_to_start(self): return self.predicted_dist_to_start
        
    
import matplotlib.pyplot as plt
def main(args):
    path = StateEstPath('/home/poine/work/overlay_ws/src/two_d_guidance/paths/demo_z/track_trr_real.npz')
    plt.plot(path.dists)
    plt.figure()
    plt.plot(path.points[:,0])
    plt.figure()
    plt.plot(path.points[:,1])
    plt.figure()
    plt.plot(path.curvatures[:])
    
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
