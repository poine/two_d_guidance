#!/usr/bin/env python

import os, sys
import math, numpy as np
import rospy, rospkg

import two_d_guidance as tdg
import two_d_guidance.trr.utils as trr_u


#
# Landmark crossing with hysteresis
#
class TrackMark:
    def __init__(self, s, name, dist=2):
        ''' landmark's abscisse, name, and monitoring distance in meters'''
        self.s, self.name = s, name
        self.monitoring_dist = dist
        self.monitor = False
        self.side = 0

        
    def update(self, dist):
        if abs(dist) > self.monitoring_dist:
            self.monitor = True
        else:
            if self.monitor:
                if dist > 0:
                    new_side = 1
                else:
                    new_side = -1
                if self.side * new_side < 0:
                    self.side = 0
                    self.monitor = False
                    print('passing lm {}: lm {:.3f} s {:.3f}'.format(self.name, self.s, dist + self.s))
                    return True
                else:
                    self.side = new_side
        return False

        
#
# Naive State estimation:
#    - integrate linear vel to predict abscice on path
#    - use landmarks (start and finish lines for now) as measurements
#

#TODO INITIALIZATION

class StateEstimator:
    
    def __init__(self, path_fname, lm_passed_cbk=None):
        self.lm_passed_cbk = lm_passed_cbk
        self.load_path(path_fname)
        self.s, self.sn, self.idx_sn, self.v = 0., 0., 0, 0 # abscice, normalized abscice, abscice idx, velocity
        self.k_odom = 1.
        self.lm_gain = 0.075
        self.lm_pred = np.full(self.path.LM_NB, float('inf'), dtype=np.float32)
        self.lm_meas = np.full(self.path.LM_NB, float('inf'), dtype=np.float32)
        self.lm_res  = np.full(self.path.LM_NB, float('inf'), dtype=np.float32)
        self.meas_dist_to_start, self.meas_dist_to_finish = float('inf'), float('inf')
        self.predicted_dist_to_start, self.predicted_dist_to_finish = float('inf'), float('inf')
        self.start_residual, self.finish_residual = float('inf'), float('inf')
        self.start_track = TrackMark(self.path.lm_s[self.path.LM_START], 'start', dist=2)
        self.finish_track = TrackMark(self.path.lm_s[self.path.LM_FINISH], 'finish', dist=2)
        self.last_stamp = None

    def load_path(self, path_filename):
        rospy.loginfo(' loading path: {}'.format(path_filename))
        self.path = trr_u.TrrPath(path_filename)
        self.path.report()

    def update_k_odom(self, v): print('k_odom {}'.format(v)); self.k_odom = v

    def update_k_lm(self, _v): self.lm_gain = _v
    
    def initialize(self, s0):
        self.sn = s0
        self.idx_sn, _ = self.path.find_point_at_dist_from_idx(0, _d=self.sn)

    # increments abscisse taking care of periodic rollover
    # check landmarks crossing
    def _update_s(self, ds):
        self.prev_sn = self.sn
        self.s += ds
        self.sn = self._norm_s(self.s)
        self.idx_sn, _ = self.path.find_point_at_dist_from_idx(0, _d=self.sn)
        
        if self.start_track.update(self._norm_s_err(self.sn - self.start_track.s)):
            if self.lm_passed_cbk is not None: self.lm_passed_cbk(self.path.LM_START)

        if self.finish_track.update(self._norm_s_err(self.sn - self.finish_track.s)):
            if self.lm_passed_cbk is not None: self.lm_passed_cbk(self.path.LM_FINISH)
            
        
    def update_odom(self, seq, stamp, vx, vy):
        self.v = vx
        if self.last_stamp is not None:
            dt = (stamp - self.last_stamp).to_sec()
            #print('odom dt {} vx {:.4f} vy {:.4f}'.format(dt, vx, vy))
            if dt < 0.01 or dt > 0.1:
                print('state est: out of range dt')
            else:
                ds = self.k_odom*vx*dt
                #print('update: {}'.format(ds))
                self._update_s(ds)
        self.last_stamp = stamp

    def _norm_s(self, s):
        while s > self.path.len: s -= self.path.len
        while s < 0: s += self.path.len
        return s

    def _norm_s_err(self, s_err):
        while s_err > self.path.len/2: s_err -= self.path.len
        while s_err < -self.path.len/2: s_err += self.path.len
        return s_err

    def update_landmark(self, lm_id, m, clip_res=1.):
        self.lm_meas[lm_id] = m
        self.lm_pred[lm_id] = self._norm_s(self.path.lm_s[lm_id]-self.sn)
        if not math.isinf(self.lm_meas[lm_id]):
            self.lm_res[lm_id] = self._norm_s_err(self.lm_meas[lm_id] - self.lm_pred[lm_id])
            self._update_s(-self.lm_gain*np.clip(self.lm_res[lm_id], -clip_res, clip_res))
        else:
            self.lm_res[lm_id] = float('inf')
            
    def update_landmarks(self, meas_dist_to_start, meas_dist_to_finish, disable_correction=False):

        #print('meas start {} meas finish {}'.format(meas_dist_to_start, meas_dist_to_finish))
        self.meas_dist_to_start, self.meas_dist_to_finish = meas_dist_to_start, meas_dist_to_finish

        self.update_landmark(self.path.LM_FINISH, meas_dist_to_finish)
        self.predicted_dist_to_finish = self.lm_pred[self.path.LM_FINISH]
        #if self.lm_res[self.path.LM_FINISH] == float('inf'): self.predicted_dist_to_finish = float('inf')
        
        self.update_landmark(self.path.LM_START, meas_dist_to_start)
        self.predicted_dist_to_start = self.lm_pred[self.path.LM_START]
        #if self.lm_res[self.path.LM_START] == float('inf'): self.predicted_dist_to_start = float('inf')
            
    
    def status(self): return self.sn, self.v

    def dist_to_start(self): return self.predicted_dist_to_start
        
