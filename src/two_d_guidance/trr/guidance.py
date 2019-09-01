import numpy as np
import rospy

import two_d_guidance as tdg
import two_d_guidance.trr.utils as trr_u

#
# This is Guidance.
# It is responsible for driving along the track
#

# Velocity control
#
# We have 3 modes:
#   -constant:
#   -profile:  depends on abscisse
#   -curv:
class VelCtl:
    mode_cst, mode_profile, mode_curv, mode_nb = range(4)
    def __init__(self, path_fname):
        self.load_profile(path_fname)
        self.mode = VelCtl.mode_cst#VelCtl.mode_curv#
        self.sp = 1.
        # curv mode
        self.min_sp = 0.2
        self.k_curv = 0.5
        
        self.ref = tdg.utils.SecOrdLinRef(omega=4., xi=0.9)
        
    def load_profile(self, path_fname):
        self.path = trr_u.TrrPath(path_fname)

        
    def get(self, lane_model, s, _is, dt=1./30):
        if self.mode == VelCtl.mode_cst:
            return self.sp
        elif self.mode == VelCtl.mode_profile:
            profile_sp = self.path.vels[_is]
            profile_acc = self.path.accels[_is]
            profile_jerk = self.path.jerks[_is]
            vel_sp = self.ref.run(dt, profile_sp)[0]
            #return self.path.vels[_is]
            return vel_sp
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
    def get_dist(self, _v):
        d1 = _v*self.t
        if d1 < 0.5: d1=0.5
        if d1 > 0.9: d1=0.9
        return d1
        
class Guidance:
    mode_idle, mode_stopped, mode_driving, mode_nb = range(4)
    mode_lookahead_dist, mode_lookahead_time = range(2)
    def __init__(self, lookahead=0.4, vel_sp=0.2, path_fname=None):
        self.set_mode(Guidance.mode_idle)
        self.lookaheads = [CstLookahead(), TimeCstLookahead()]
        self.lookahead_mode = Guidance.mode_lookahead_dist
        self.lookahead_dist = lookahead
        self.lookahead_time = 0.1
        self.lane = trr_u.LaneModel()
        self.carrot = [self.lookahead_dist, 0]
        self.R = np.inf
        self.vel_ctl = VelCtl(path_fname)
        self.vel_sp = vel_sp
        self.lin_sp, self.ang_sp = 0, 0

    
    def compute(self, s=float('inf'), _is=0, expl_noise=0.025, dy=0., avoid_obstacles=False):
        if self.mode == Guidance.mode_driving and self.lane.is_valid():
            lin = self.vel_ctl.get(self.lane, s, _is)
            if avoid_obstacles:
                if s > 15.5 and s < 16.: dy += 0.25
                elif s > 0.5 and s < 1.: dy -= 0.25
            self.lookahead_dist = self.lookaheads[self.lookahead_mode].get_dist(lin)
            self.lookahead_time = np.inf if lin == 0 else self.lookahead_dist/lin
            self.carrot = [self.lookahead_dist, self.lane.get_y(self.lookahead_dist)+dy]
            self.R = (np.linalg.norm(self.carrot)**2)/(2*self.carrot[1])
            lin, ang = lin, lin/self.R
            ang += expl_noise*np.sin(0.5*rospy.Time.now().to_sec())
        else:
            lin, ang = 0., 0.
        self.lin_sp, self.ang_sp = lin, ang
        return lin, ang

    def set_mode(self, mode):
        rospy.loginfo('guidance: setting mode to {}'.format(mode))
        self.mode = mode
    