import numpy as np

def normalize_headings2(h):
    # normalize angles to [-pi; pi[
    return ( h + np.pi) % (2 * np.pi ) - np.pi

def normalize_headings(h):
    # normalize angles to ]-pi; pi]
    return -((np.pi - h) % (2 * np.pi ) - np.pi)

def pt_on_circle(c, r, th):
    return c + np.stack([r*np.cos(th), r*np.sin(th)], axis=-1)



#
#  Linear reference models
#

class LinRef:
    ''' Linear Reference Model (with first order integration)'''
    def __init__(self, K):
        '''K: coefficients of the caracteristic polynomial, in ascending powers order,
              highest order ommited (normalized to -1)'''
        self.K = K; self.order = len(K)
        self.X = np.zeros(self.order+1)

    def run(self, dt, sp, max_accel=None):
        self.X[:self.order] += self.X[1:self.order+1]*dt
        e =  np.array(self.X[:self.order]); e[0] -= sp
        # FIXME - cliping of highest order derivative
        if max_accel is not None: e[-2] = np.clip(e[-2], -max_accel, max_accel)
        self.X[self.order] = np.sum(e*self.K)
        return self.X

    def poles(self):
        return np.roots(np.insert(np.array(self.K[::-1]), 0, -1))

    def reset(self, X0=None):
        if X0 is None: X0 = np.zeros(self.order+1)
        self.X = X0


class FirstOrdLinRef(LinRef):
    def __init__(self, tau):
        LinRef.__init__(self, [-1/tau])

class SecOrdLinRef(LinRef):
    def __init__(self, omega, xi):
        LinRef.__init__(self, [-omega**2, -2*xi*omega])



