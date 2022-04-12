
import sys , math, numpy as np

import two_d_guidance.path
import pdb

class Param:
    def __init__(self):
        self.L = 1.

class EndOfPathException(Exception):
    pass

class PurePursuit:
    mode_idle, mode_stopped, mode_driving, mode_nb = range(4)
    def __init__(self, path_file, params, look_ahead=0.3):
        self.path = two_d_guidance.path.Path(load=path_file)
        self.params = params
        self.look_ahead = look_ahead
        self.p2, self.R = [0, 0], 1e6
        self.mode = PurePursuit.mode_idle
        
    def set_mode(self, m): self.mode = m
    def set_look_ahead_dist(self, l): self.look_ahead = l

    def set_path(self, path):
        self.path = path
        self.path.reset()
 
    def compute(self, cur_pos, cur_psi, looped=True):
        if looped:
            p1, p2, end_reached, ip1, ip2 = self.path.find_carrot_looped(cur_pos, _d=self.look_ahead)
        else:
            p1, p2, end_reached, ip1, ip2 = self.path.find_carrot_alt(cur_pos, _d=self.look_ahead)
            
        self.p2 = p2
        if end_reached:
            raise EndOfPathException

        p0p2_w = p2 - cur_pos
        cy, sy = math.cos(cur_psi), math.sin(cur_psi)
        w2b = np.array([[cy, sy],[-sy, cy]])
        p0p2_b = np.dot(w2b, p0p2_w)
        l = np.linalg.norm(p0p2_w)
        R = (l**2)/(2*p0p2_b[1])
        self.R = R
        return 0, math.atan(self.params.L/R)
        #return R, p2 # Radius and carrot 

    def compute_looped(self, cur_pos, cur_psi):
        try:
            return self.compute(cur_pos, cur_psi)
        except EndOfPathException:
            self.path.reset()
            return self.compute(cur_pos, cur_psi)
