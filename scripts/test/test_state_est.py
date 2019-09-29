#!/usr/bin/env python
import sys, time, numpy as np
import rospy
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

import two_d_guidance as tdg
import two_d_guidance.trr.state_estimation as trr_se
import two_d_guidance.trr.utils as trr_u
import two_d_guidance.plot_utils as pu
import pdb    


def plot_lane_coefs(time, lane_coefs):
    fig, axs = plt.subplots(2, 1)
    for i in range(lane_coefs.shape[1]):
        axs[i].plot(lane_coefs[:,i])
        pu.decorate(axs[i],'$a_{}$'.format(3-i), xlab="steps")

def plot_chronogram(lane_times, cs, odom_times, curv_odom_times, path_curv_shifted_odom_times, distn_odom_times):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(lane_times, cs, '.-', label='vision')
    #axs[0].plot(odom_times, curv_odom_times, label='path')
    axs[0].plot(odom_times, path_curv_shifted_odom_times, label='path shifted')
    pu.decorate(axs[0], 'curvature', xlab="time in s", ylab='c in m-1', legend=True)
    #axs[1].plot(odom_times, distn_odom_times, label='vision')
    #pu.decorate(axs[1], 'ndists', xlab="time in s", ylab='sn in m', legend=True)
 
def plot_run(times, lane_mod_coefs, Cs, cs, p):
    plt.subplot(4,1,1)
    plt.plot(cs, label='vision'); plt.title('curvature vision')
    #plt.plot(p.curvatures); plt.title('curvature path')
    plt.subplot(4,1,2)
    plt.plot(Cs[:,0])
    plt.subplot(4,1,3)
    plt.plot(Cs[:,1])
    plt.subplot(4,1,4)
    plt.plot(p.curvatures); plt.title('curvature path')
    if 0:
        plots = [("curvature", "m-1", cs), ("$xc$", "m", Cs[:,0]), ("$yc$", "m", Cs[:,1]), ("$dist$", "m", dists)]
        return pu.plot_in_grid(odom_times, plots, 2, figure, window_title, legend, filename,
                               margins=(0.04, 0.08, 0.99, 0.92, 0.14, 0.51))


import scipy.optimize
def pt_on_circle(xc, yc, R, thetas): return np.stack([xc+R*np.cos(thetas), yc+R*np.sin(thetas)], axis=1)

def compute_curvature(lane_coefs, x_min, x_max, c0, xc0, yc0): return compute_curvature2(lane_coefs, x_min, x_max, c0, xc0, yc0)
#
def compute_curvature1(lane_coefs, x_min, x_max, c0, xc0, yc0):
        xps = np.linspace(x_min, x_max)
        yps = np.polyval(lane_coefs, xps)
        if lane_coefs[0] < 0:
            c0, yc0 =-c0, -yc0

        def residual(p):
            xc, yc, c = p
            if True:#c != 0: # FIXME, epsilon?
                thetas = np.arctan2(yps-yc, xps-xc)
                #pdb.set_trace()
                #print thetas
                Xcs = pt_on_circle(xc, yc, np.abs(1./c), thetas)
            else:
                Xcs = np.stack([xps, np.zeros(len(xps))], axis=1)
            Xps = np.stack([xps, yps], axis=1)
            return np.linalg.norm(Xps-Xcs, axis=1)
        p0 = [xc0, yc0, c0]
        popt, pcov = scipy.optimize.leastsq(residual, p0)
        xc, yc, c = popt
        return [xc, yc], c
#
# https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
#
def compute_curvature2(lane_coefs, x_min, x_max, c0, xc0, yc0):
    xps = np.linspace(x_min, x_max)
    yps = np.polyval(lane_coefs, xps)

    #def calc_R(xc, yc, xs, ys): return np.sqrt((xs-xc)**2 + (ys-yc)**2)
    def calc_Rs(xc, yc): return np.sqrt((xps-xc)**2 + (yps-yc)**2)
    def residual(p):
        xc, yc = p
        Ri = calc_Rs(xc, yc)
        return Ri - Ri.mean()
    if lane_coefs[0] < 0:
        yc0 = -yc0
    p0 = [xc0, yc0]
    popt, pcov = scipy.optimize.leastsq(residual, p0)
    #pdb.set_trace()
    xc, yc = popt
    R = calc_Rs(xc, yc).mean()
    if lane_coefs[0] < 0:
             R = -R
    return [xc, yc], 1./R
    
def compute_curvatures(lane_mod_coefs, x_min, x_max):
    nb_frame = len(lane_mod_coefs)
    Cs, cs = np.zeros((nb_frame, 2)), np.zeros(nb_frame)
    c0, xc0, yc0 = 1./1.75, 0.4, 1.75
    for i, lmc in enumerate(lane_mod_coefs):
        #if i > 0: c0, xc0, yc0 = cs[i-i], Cs[i-1,0], Cs[i-1,1]
        Cs[i], cs[i] = compute_curvature(lmc, x_min, x_max, c0, xc0, yc0)
    return Cs, cs

def plot_frame(frame_id, lane_mod_coefs, C=None, c=None):
    x_min, x_max = 0.3, 1.5
    xps = np.linspace(x_min, x_max)
    yps = np.polyval(lane_mod_coefs[frame_id], xps)
    plt.plot(xps, yps, label='poly')
    if C is not None:
        xc, yc = C
        thetas = np.arctan2(yps-yc, xps-xc)
        Xcs = pt_on_circle(xc, yc, np.abs(1/c), thetas)
        plt.plot(Xcs[:,0], Xcs[:,1], label='circle')
    ax = plt.gca()
    ax.axis('equal')
    pu.decorate(ax,'frame {}'.format(frame_id), xlab="x in m", ylab='y in m', legend=True)
    if 0:
        plot_extents = 0, 10, 0, 10
        transform = Affine2D().rotate_deg(45)
        helper = floating_axes.GridHelperCurveLinear(transform, plot_extents)
        ax = floating_axes.FloatingSubplot(plt.gcf, 111, grid_helper=helper)
    
        

def plot_frames(f0, f1, lane_mod_coefs, Cs, cs):
    print('plotting frames: {} to {} '.format(f0, f1))
    for frame_id in range(f0, f1):
        plot_frame(frame_id, lane_mod_coefs, Cs[frame_id], cs[frame_id])
        plt.savefig('/tmp/lanes/frame_{:06d}.png'.format(frame_id))
        plt.clf()

def plot_path(p):
    tdg.draw_path(plt.gcf(), plt.gca(), p)
    plt.figure()
    tdg.draw_path_curvature(plt.gcf(), plt.gca(), p)
    plt.show()

def plot_odom(odom_times, odom_vlins, odom_vangs, figure=None, window_title="Reference", legend=None, filename=None):
    plots = [("$lin vel$", "m/s", odom_vlins), ("$ang_vel$", "m", odom_vangs)]
    fig = pu.plot_in_grid(odom_times, plots, 2, figure, window_title, legend, filename,
                           margins=(0.04, 0.08, 0.99, 0.92, 0.14, 0.51))
    return fig

        

def get_vision_measurements(lane_mod_coefs, force_recompute=False, filename='/tmp/est_run.npz'):
    if force_recompute:
        Cs, cs = compute_curvatures(lane_mod_coefs, 0.3, 1.5)
        print('saving est run to {}'.format(filename))
        np.savez(filename, Cs=Cs, cs=cs)
    else:
        print('loading est run from {}'.format(filename))
        data =  np.load(filename)
        Cs, cs = data['Cs'], data['cs']
    return Cs, cs

def test_state_est(d0, odom_times, odom_vlins, path_fname):
    dists = np.zeros(len(odom_times))
    dists[0] = d0
    for i in range(1, len(odom_times)):
        dists[i] = dists[i-1] + (odom_times[i]-odom_times[i-1])*(odom_vlins[i-1]+odom_vlins[i])/2

    path_curv_odom_times = np.zeros(len(odom_times))
    path_curv_shifted_odom_times = np.zeros(len(odom_times))
    distn_odom_times = np.zeros(len(odom_times)) 
    se = trr_se.StateEstimator(path_fname)
    se.initialize(d0)
    for i, (t, vx) in enumerate(zip(odom_times, odom_vlins)):
        se.update_odom(0, rospy.Time.from_sec(t), vx, 0)
        path_curv_odom_times[i]  = -se.path.curvatures[se.idx_sn] # FIXME, why -?
        idx_sn_delay, _ = se.path.find_point_at_dist_from_idx(0, _d=se.sn-d0)
        #pdb.set_trace()
        try:
            path_curv_shifted_odom_times[i]  = -se.path.curvatures[idx_sn_delay]
        except ValueError:
            #print(idx_sn_delay, se.path.curvatures[idx_sn_delay])
            pass
        distn_odom_times[i] = se.sn
        
    return dists, path_curv_odom_times, path_curv_shifted_odom_times, distn_odom_times


def test_vision_curvature(lane_coefs, idx):
    # if lane_coefs[idx, 0] < 0:
    #     C, c = compute_curvature(lane_coefs[idx], 0.3, 1.2, c0=-1./1.75, xc0=0.4, yc0=-1.75)
    # else:
    #     C, c = compute_curvature(lane_coefs[idx], 0.3, 1.2, c0=1./1.75, xc0=0.4, yc0=1.75)
    C, c = compute_curvature(lane_coefs[idx], 0.3, 1.2, c0=1./1.75, xc0=0.4, yc0=1.75)
    print('test vision curvature {} {} (R={:.1f})'.format(C, c, 1./c if c!=0 else nop.float('inf')))
    plot_frame(idx, lane_coefs, C, c)
    plt.show()

def process_vision_log(vision_pipe_filename='/tmp/pipe_run.npz'):
    #path_fname = '/home/poine/work/overlay_ws/src/two_d_guidance/paths/demo_z/track_trr_real.npz'
    path_fname = '/home/poine/work/overlay_ws/src/two_d_guidance/paths/vedrines/track_trr_0.npz'
    _path = trr_u.TrrPath(path_fname)
    #plot_path(p)
    print('loading vision log  from {}'.format(vision_pipe_filename))
    data =  np.load(vision_pipe_filename)
    lane_times, lane_mod_coefs = data['times'], data['lane_mod_coefs']
    odom_times, odom_vlins, odom_vangs = data['odom_times'], data['odom_vlins'], data['odom_vangs']

    if 0:
        plot_lane_coefs(lane_times, lane_mod_coefs)
        plt.show()

    if 0:
        plot_odom(odom_times, odom_vlins, odom_vangs)

    if 0:
        #test_vision_curvature(lane_mod_coefs, 390)
        #test_vision_curvature(lane_mod_coefs, 601)
        #test_vision_curvature(lane_mod_coefs, 756)
        test_vision_curvature(lane_mod_coefs, 1300)
        plt.show()
    Cs, cs = get_vision_measurements(lane_mod_coefs, force_recompute=True)
    if 0:
        plot_frames(0, len(lane_mod_coefs), lane_mod_coefs, Cs, cs)
    
    dists, curv_odom_times, path_curv_shifted_odom_times, distn_odom_times = test_state_est(-1, odom_times, odom_vlins, path_fname)
    #plot_run(lane_times, lane_mod_coefs, Cs, cs, _path)
    plot_chronogram(lane_times, cs, odom_times, curv_odom_times, path_curv_shifted_odom_times, distn_odom_times)
    plt.show()

    
def main(args):
    #plot_path()
    process_vision_log()
      
if __name__ == '__main__':
    main(sys.argv)
