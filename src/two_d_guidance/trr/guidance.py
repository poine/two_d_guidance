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
        self.sp = 1.5
        # curv mode
        self.min_sp = 1.
        self.k_curv = 0.5
        # second order reference model driven by input setpoint
        _sats = [4., 25.]  # accel, jerk
        self.ref = tdg.utils.SecOrdLinRef(omega=6., xi=0.9, sats=_sats)
        
    def load_profile(self, path_fname):
        self.path = trr_u.TrrPath(path_fname)
        self.path.report()

    def reset_ref(self, v0):
        self.ref.reset(np.array([v0, 0, 0]))
        
    def get(self, lane_model, s, _is, dt=1./30):
        if self.mode == VelCtl.mode_cst:
            #return self.sp # unfiltered
            vel_ref = self.ref.run(dt, self.sp)[0]
            return vel_ref
        elif self.mode == VelCtl.mode_profile:
            profile_sp = self.path.vels[_is]
            profile_acc = self.path.accels[_is]
            profile_jerk = self.path.jerks[_is]
            vel_ref = self.ref.run(dt, profile_sp)[0]
            #return self.path.vels[_is]
            return vel_ref
        else:
            curv = lane_model.coefs[1]
            return max(self.min_sp, self.sp-np.abs(curv)*self.k_curv)

class CstLookahead:
    def __init__(self, _d=0.4): self.d = _d
    def set_dist(self, _d): self.d = _d
    #def set_time(self, _t): pass
    def get_dist(self, _v): return self.d

class AdaptiveLookahead:
    def __init__(self):
        self.v0, self.v1 = 2., 4.
        self.d0, self.d1 = 1.2, 2.0
        self.k =  (self.d1-self.d0)/(self.v1-self.v0)
    def set_dist(self, _d): pass
    def set_time(self, _t): pass #self.t = _t
    def get_dist(self, _v):
        if _v < self.v0:   return self.d0
        elif _v > self.v1: return self.d1
        else: return self.d0 + (_v-self.v0)*self.k
        
class Guidance:
    mode_idle, mode_stopped, mode_driving, mode_nb = range(4)
    mode_lookahead_cst, mode_lookahead_adaptive = range(2)
    def __init__(self, lookahead=0.4, vel_sp=0.2, path_fname=None):
        self.lookaheads = [CstLookahead(), AdaptiveLookahead()]
        self.lookahead_mode = Guidance.mode_lookahead_cst
        self.lookahead_dist = lookahead
        self.lookahead_time = 0.1
        self.lane = trr_u.LaneModel()
        self.carrot = [self.lookahead_dist, 0]
        self.R = np.inf
        self.vel_ctl = VelCtl(path_fname)
        self.vel_sp = vel_sp
        self.lin_sp, self.ang_sp = 0, 0
        self.est_vel = 0.
        self.set_mode(Guidance.mode_idle)
    
    def compute(self, s, _is, est_vel, expl_noise=0.025, dy=0., avoid_obstacles=False):
        self.est_vel = est_vel
        if self.mode == Guidance.mode_driving and self.lane.is_valid():
            lin = self.vel_ctl.get(self.lane, s, _is)
            if avoid_obstacles:
                if s > 15.5 and s < 16.: dy += 0.25
                elif s > 0.5 and s < 1.: dy -= 0.25
            self.lookahead_dist = self.lookaheads[self.lookahead_mode].get_dist(lin)
            self.lookahead_time = np.inf if lin == 0 else self.lookahead_dist/lin
            delay = rospy.Time.now().to_sec() - self.lane.stamp.to_sec()
            self.carrot = [self.lookahead_dist, self.lane.get_y(self.lookahead_dist)+dy]
            #self.carrot = _time_compensate(self.carrot, self.lin_sp, self.ang_sp, delay=delay)
            self.R = (np.linalg.norm(self.carrot)**2)/(2*self.carrot[1])
            lin, ang = lin, lin/self.R
            ang += expl_noise*np.sin(0.5*rospy.Time.now().to_sec())
        else:
            lin, ang = 0., 0.
        self.lin_sp, self.ang_sp = lin, ang
        return lin, ang

    def set_mode(self, mode):
        rospy.loginfo('guidance: setting mode to {} {}'.format(mode, self.est_vel))
        self.mode = mode
        self.vel_ctl.reset_ref(self.est_vel)

        
    def load_vel_profile(self, path_filename):
        res = self.vel_ctl.load_profile(path_filename)
        return res


def _time_compensate(carrot, previous_speed, previous_ang, delay=1/30):
    if previous_ang == 0 or previous_speed == 0:
        carrot[0] -= previous_speed * delay
        return carrot
    else:
        # Arc length = R * angle
        # =>(1) angle = arc length / R
        # Arc length = speed * time
        # =>(2) arc length = previous_speed * delay
        # (3) R = previous_speed / previous_ang
        # (1, 2, 3) => angle = previous_speed * delay / (previous_speed / previous_ang)
        # => angle = delay * previous_ang
        rotation_ang = delay * previous_ang
        previous_r = previous_speed / previous_ang
        carrot[0] -= np.sin(rotation_ang)*previous_r
        carrot[1] += (1 - np.cos(rotation_ang)) * previous_r
        new_x = carrot[0] * np.cos(rotation_ang) + carrot[1] * np.sin(rotation_ang)
        new_y = -carrot[0] * np.sin(rotation_ang) + carrot[1] * np.cos(rotation_ang)
        return [new_x, new_y]

