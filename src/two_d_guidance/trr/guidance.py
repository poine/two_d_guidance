import numpy as np
import rospy

import two_d_guidance.trr.utils as trr_u

class VelCtl:
    mode_cst, mode_profile, mode_curv, mode_nb = range(4)
    def __init__(self, path_fname):
        self.path = trr_u.TrrPath(path_fname)
        self.mode = VelCtl.mode_cst#VelCtl.mode_curv#
        self.sp = 1.
        self.min_sp = 0.2
        self.k_curv = 0.5

    def get(self, lane_model, s, _is):
        if self.mode == VelCtl.mode_cst:
            return self.sp
        elif self.mode == VelCtl.mode_profile:
            #_is, _ = self.path.find_point_at_dist_from_idx(0, _d=s)
            #print s, _is
            return self.path.vels[_is]
        else:
            curv = lane_model.coefs[1]
            return max(self.min_sp, self.sp-np.abs(curv)*self.k_curv)

class CstLookahead:
    def __init__(self, _d=0.4): self.d = _d
    def set_dist(self, _d): self.d = _d
    def set_time(self, _t): pass
    def get_dist(self, _v): return self.d

class TimeCstLookahead:
    def __init__(self, _t=0.4): self.t = _t
    def set_dist(self, _d): pass
    def set_time(self, _t): self.t = _t
    def get_dist(self, _v): return _v*self.t
        
class Guidance:
    mode_idle, mode_stopped, mode_driving, mode_nb = range(4)
    mode_lookahead_dist, mode_lookahead_time = range(2)
    def __init__(self, lookahead=0.4, vel_sp=0.2, path_fname=None):
        self.set_mode(Guidance.mode_idle)
        self.lookaheads = [CstLookahead(), TimeCstLookahead()]
        self.lookahead_mode = Guidance.mode_lookahead_dist
        self.lookahead_dist = 0.1
        self.lookahead_time = 0.1
        self.carrot = [self.lookahead_dist, 0]
        self.R = np.inf
        self.vel_ctl = VelCtl(path_fname)
        self.vel_sp = vel_sp
    
    def compute(self, lane_model, s=float('inf'), _is=0, expl_noise=0.025, dy=0., avoid_obstacles=False):
        lin = self.vel_ctl.get(lane_model, s, _is)
        if avoid_obstacles:
            if s > 15.5 and s < 16.: dy += 0.25
            elif s > 0.5 and s < 1.: dy -= 0.25
        self.lookahead_dist = self.lookaheads[self.lookahead_mode].get_dist(lin)
        self.lookahead_time = np.inf if lin == 0 else self.lookahead_dist/lin
        self.carrot = [self.lookahead_dist, lane_model.get_y(self.lookahead_dist)+dy]
        self.R = (np.linalg.norm(self.carrot)**2)/(2*self.carrot[1])
        lin, ang = lin, lin/self.R
        ang += expl_noise*np.sin(0.5*rospy.Time.now().to_sec())
        return lin, ang

    def set_mode(self, mode):
        rospy.loginfo('guidance: setting mode to {}'.format(mode))
        self.mode = mode
    
