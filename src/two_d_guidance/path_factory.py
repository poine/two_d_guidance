#!/usr/bin/env python
import logging, sys, math, numpy as np
import matplotlib.pyplot as plt, matplotlib

from. import path, utils, plot_utils

import pdb



def pt_on_ellipse(a, b, th):
    return np.stack([a*np.cos(th), b*np.sin(th)], axis=-1)

def make_line_path(p0, p1, n_pt=10, res=None):
    _path = path.Path()
    if res is not None:
        disp = np.array(p1)-p0
        dist = np.linalg.norm(disp)
        #pdb.set_trace()
        n_pt = int(dist/res)
        #print(' npt {}'.format(n_pt))
    _pts = np.stack([np.linspace(p0[i], p1[i], n_pt) for i in range(2)], axis=-1)
    disp = np.asarray(p1)-p0
    headings = np.arctan2(disp[1], disp[0])*np.ones(len(_pts))
    dists = np.linspace(0, np.linalg.norm(disp), n_pt)
    curvatures = np.zeros(n_pt)
    _path.append_points(_pts, headings, dists, curvatures)
    return _path

def make_circle_path(c, r, th0, dth, n_pt=10, res=None):
    _path = path.Path()
    if res is not None:
        dist = abs(dth*r)
        n_pt = int(dist/res)
    ths = np.linspace(th0, th0+np.sign(r)*dth, n_pt, endpoint=False)
    points = utils.pt_on_circle(c, np.abs(r), ths)
    headings = utils.normalize_headings(ths + np.sign(r)*np.pi/2)
    curvatures = 1./r*np.ones(n_pt)
    dists = np.linspace(0, np.abs(r*dth), n_pt)
    _path.append_points(points, headings, dists, curvatures)
    return _path

def make_spirale_path(c, r0, th0, dth, dr, n_pt=10):
    _path = path.Path()
    ths = np.linspace(th0, th0+dth, n_pt, endpoint=False)
    rs = np.linspace(r0, r0+dr, n_pt, endpoint=False)
    points = utils.pt_on_circle(c, rs, ths)
    headings = np.zeros(n_pt) # TODO
    dists = np.zeros(n_pt) # TODO
    curvatures = np.zeros(n_pt) # TODO
    _path.append_points(points, headings, dists, curvatures)
    _path.compute_headings()
    _path.compute_dists()
    _path.compute_curvatures()
    #pdb.set_trace()
    return _path

## CHECKME
def make_ellipse_path(f1, f2, d, th0=0., th1=2*np.pi, n_pt=360, cw=False):
    a = d/2
    c = np.linalg.norm(np.asarray(f1)-f2)/2
    b = np.sqrt(a**2-c**2)
    C = (np.asarray(f1)+f2)/2
    #pdb.set_trace()
    _path = path.Path()
    ths = np.linspace(th0, th1, n_pt)
    if cw:
        b = -b
    pts = C + pt_on_ellipse(a, b, ths)
    headings = np.arctan2(b*np.cos(ths), -a*np.sin(ths))
    dists = np.zeros(n_pt) # TODO
    curvatures = np.zeros(n_pt) # TODO
    _path.append_points(pts, headings, dists, curvatures=curvatures)
    _path.compute_dists()
    _path.compute_curvatures()
    return _path

def make_oval_path(c1, c2, r):
    axis = np.array(c1)-c2
    angle = np.arctan2(axis[1], axis[0])
    if r > 0:
        circle1 = make_circle_path(c1, r, angle-np.pi/2, np.pi, n_pt=180)
        circle2 = make_circle_path(c2, r, angle+np.pi/2, np.pi, n_pt=180)
        line1 = make_line_path(circle2.points[-1], circle1.points[0], n_pt=100)
        line2 = make_line_path(circle1.points[-1], circle2.points[0], n_pt=100)
        line1.append([circle1, line2, circle2])
    else:
        circle1 = make_circle_path(c1, r, angle+np.pi/2, np.pi, n_pt=180)
        circle2 = make_circle_path(c2, r, angle-np.pi/2, np.pi, n_pt=180)
        line1 = make_line_path(circle1.points[-1], circle2.points[0], n_pt=100)
        line2 = make_line_path(circle2.points[-1], circle1.points[0], n_pt=100)
        line1.append([circle2, line2, circle1])
    return line1


def make_fig_of_eight_path2(c, d, r):
    c1 = c + [0, d]
    circle1 = make_circle_path(c1, -r, -math.pi, 0, n_pt=100)
    c2 = c + [0, -d]
    circle2 = make_circle_path(c2, -r,  math.pi, 0, n_pt=100)
    line1 = make_line_path(circle1.points[-1], circle2.points[0], n_pt=100)
    line2 = make_line_path(circle2.points[-1], circle1.points[0], n_pt=100)
    circle1.append([line1, circle2, line2])
    return circle1

def make_fig_of_eight_path2(r):
    p = make_circle_path([r, 0], r, -np.pi, 2*np.pi, 360)
    p2 = make_circle_path([-r, 0], -r, 0, 2*np.pi, 360)
    p.append([p2])
    return p
    
def make_clover_path(r1=1., r2=2.):
    c1 = [r1+r2, 0]
    circ1 = make_circle_path(c1, -r1, -0.5*np.pi, np.pi, n_pt=180)
    c2 = [r1+r2, r1+r2]
    circ2 = make_circle_path(c2, r2, -0.5*np.pi, 1.5*np.pi, n_pt=270)
    c3 = [0, r1+r2]
    circ3 = make_circle_path(c3, -r1, 0, np.pi, n_pt=180)
    c4 = [-(r1+r2), r1+r2]
    circ4 = make_circle_path(c4, r2, 0, 1.5*np.pi, n_pt=270)
    c5 = [-(r1+r2), 0]
    circ5 = make_circle_path(c5, -r1, 0.5*np.pi, np.pi, n_pt=180)
    c6 = [-(r1+r2), -(r1+r2)]
    circ6 = make_circle_path(c6, r2, 0.5*np.pi, 1.5*np.pi, n_pt=270)
    c7 = [0, -(r1+r2)]
    circ7 = make_circle_path(c7, -r1, np.pi, np.pi, n_pt=180)
    c8 = [r1+r2, -(r1+r2)]
    circ8 = make_circle_path(c8, r2, -np.pi, 1.5*np.pi, n_pt=270)
    circ1.append([circ2, circ3, circ4, circ5, circ6, circ7, circ8])
    return circ1


def make_el_fig_of_eight_path(d1, d2):
    #p = make_ellipse_path([-12.6, 0], [-0.6, 0], 13.2, 0, 2*np.pi, cw=False)
    #p2 = make_ellipse_path([0.6, 0], [12.6, 0], 13.2, -np.pi, np.pi, cw=True)
    p = make_ellipse_path([-d1-d2, 0], [-d2, 0], d1+2*d2, 0, 2*np.pi, cw=False)
    p2 = make_ellipse_path([d2, 0], [d1+d2, 0], d1+2*d2, -np.pi, np.pi, cw=True)
    p.append([p2])
    return p


def make_inscribed_circles_path(r1=1, r2=2):
    c1 = make_circle_path([0, 0], r2, -np.pi, np.pi, n_pt=180)
    c2 = make_circle_path([r1+r2, 0], -r1, -np.pi, np.pi, n_pt=180)
    r4 = 2*r1+r2
    c3 = make_circle_path([0, 0], -r4, 0, 0.5*np.pi, n_pt=90)
    r3 = (2*r1+r2)/2
    c4 = make_circle_path([0, -r3], -r3, -0.5*np.pi, np.pi, n_pt=180)
    c5 = make_circle_path([0, r3], r3, -0.5*np.pi, np.pi, n_pt=180)
    c6 = make_circle_path([0, 0], r4, 0.5*np.pi, 0.5*np.pi, n_pt=90)
    c7 = make_circle_path([-(r1+r2), 0], r1, -np.pi, np.pi, n_pt=180)
    c8 = make_circle_path([0, 0], -r2, -np.pi, np.pi, n_pt=180)
    c9 = make_circle_path([r1+r2, 0], r1, -np.pi, np.pi, n_pt=180)
    c10 = make_circle_path([0, 0], r4, 0, 0.5*np.pi, n_pt=90)
    c11 = make_circle_path([0, r3], r3, 0.5*np.pi, np.pi, n_pt=180)
    c12 = make_circle_path([0, -r3], -r3, 0.5*np.pi, np.pi, n_pt=180)
    c13 = make_circle_path([0, 0], -r4, -0.5*np.pi, 0.5*np.pi, n_pt=90)
    c14 = make_circle_path([-(r1+r2), 0], -r1, -np.pi, np.pi, n_pt=180)
    c1.append([c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14])
    return c1


def make_reversed_path(_pf):
    _points = _pf.points[::-1]
    _dists = _pf.dists[-1]-_pf.dists[::-1]
    _headings = _pf.headings[::-1] + np.pi # todo: Normalize
    _curvatures = -_pf.curvatures[::-1]
    _pr = path.Path()
    _pr.append_points(_points, _headings, _dists, _curvatures)
    #pdb.set_trace()
    return _pr



def check_curvature(p):
    orig_curv = np.array(p.curvatures)
    p.compute_curvatures()
    plt.figure()
    plt.plot(orig_curv)
    plt.plot(p.curvatures)
    plt.legend(['orig', 'computed'])


def draw_path(fig, ax, p):
    points = p.points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(p.points))
    lc = matplotlib.collections.LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(np.arange(len(p.points)))
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    dx, dy = 0.1, 0.1
    ax.set_xlim(p.points[:,0].min()-dx, p.points[:,0].max()+dx)
    ax.set_ylim(p.points[:,1].min()-dy, p.points[:,1].max()+dy)
    ax.set_aspect('equal'); plt.title('2D: vertex indicices along path')

def draw_path_curvature(fig, ax, path):
    plt.title('curvatures')
    plt.plot(path.curvatures)

def draw_path_vel(fig, ax, path):
    plt.title('vel')
    plt.plot(path.vels)

def draw_path_accel(fig, ax, path):
    plt.title('accel')
    plt.plot(path.accels)

def draw_path_vel_profile(fig, path):
    ax1 = plt.subplot(3,1,1)
    plt.plot(path.vels)
    plot_utils.decorate(ax1, 'vel', 'idx', 'm/s')
    ax2 = plt.subplot(3,1,2)
    plt.plot(path.accels)
    plot_utils.decorate(ax2, 'accel', 'idx', 'm/s2')
    ax3 = plt.subplot(3,1,3)
    plt.plot(path.jerks)
    plot_utils.decorate(ax3, 'jerk', 'idx', 'm/s3')

def draw_path_vel_profile_chrono(fig, path):
    ax1 = plt.subplot(3,1,1)
    plt.plot(path.time, path.vels)
    plot_utils.decorate(ax1, 'vel', 'time (s)', 'm/s')
    ax2 = plt.subplot(3,1,2)
    plt.plot(path.time, path.accels)
    plot_utils.decorate(ax2, 'accel', 'time (s)', 'm/s2')
    ax3 = plt.subplot(3,1,3)
    plt.plot(path.time, path.jerks)
    plot_utils.decorate(ax3, 'jerk', 'time (s)', 'm/s3')
    

def draw_path_vel_2D(fig, ax, path):
    draw_path_2D(fig, ax, path, path.vels)
    plt.title('2D: velocity along path')

def draw_path_accel_2D(fig, ax, path):
    draw_path_2D(fig, ax, path, path.accels)
    plt.title('2D: accels along path')

def draw_path_lat_accel_2D(fig, ax, path):
    lat_accel = np.abs(path.curvatures*np.square(path.vels))
    draw_path_2D(fig, ax, path, lat_accel)
    plt.title('2D: lat accels along path')
    

def draw_path_2D(fig, ax, path, color_vals):
    points = path.points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    min_v, max_v = np.min(color_vals), np.max(color_vals)
    norm = plt.Normalize(min_v, max_v)
    lc = matplotlib.collections.LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(color_vals)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    dx, dy = 0.1, 0.1
    ax.set_xlim(path.points[:,0].min()-dx, path.points[:,0].max()+dx)
    ax.set_ylim(path.points[:,1].min()-dy, path.points[:,1].max()+dy)
    ax.set_aspect('equal')
    
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #main(sys.argv)
    #test()
