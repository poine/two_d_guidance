#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import visualization_msgs.msg, geometry_msgs.msg
import math, numpy as np
import cv2

import two_d_guidance as tdg
import two_d_guidance.ros_utils as ros_utils
import two_d_guidance.msg


def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi

def find_extends(cnt):
    xmin, xmax = np.min(cnt[:,0]), np.max(cnt[:,0])
    ymin, ymax = np.min(cnt[:,1]), np.max(cnt[:,1])
    return xmin, xmax, ymin, ymax

def print_extends(cnt, txt, full=False):
    xmin, xmax, ymin, ymax = find_extends(cnt)
    if not full:
        print('  {}:    x {:.2f} {:.2f} y {:.2f} {:.2f}'.format(txt, xmin, xmax, ymin, ymax))
    else:
        print('  {}: len {} x {:.2f} {:.2f} y {:.2f} {:.2f}'.format(txt, len(cnt), xmin, xmax, ymin, ymax))


#
# This is a specialized version of a path including a velocity profile
#
class TrrPath(tdg.path.Path):
    def __init__(self, path_filename, v=0.6):
        tdg.path.Path.__init__(self)
        data = self.load(path_filename)
        try:
            self.vels = data['vels']
            self.accels = data['accels']
            self.jerks = data['jerks']
        except KeyError:
            print(' -no vel/acc/jerk in archive, setting them to zero')
            self.vels = v*np.ones(len(self.points))
            self.accels = v*np.ones(len(self.points))
            self.jerks = v*np.ones(len(self.points))
            
            
    def save(self, filename):
        print('saving path to {}'.format(filename))
        np.savez(filename, points=self.points, headings=self.headings, curvatures=self.curvatures, dists=self.dists, vels=self.vels, accels=self.accels, jerks=self.jerks)



class LaneModel:
    # center line as polynomial
    # y = an.x^n + ...
    def __init__(self):
        self.coefs = [0., 0., 0., 0.01, 0.05]
        self.valid = False

    def is_valid(self): return self.valid
    def set_valid(self, v): self.valid = v
        
    def get_y(self, x):
        return np.polyval(self.coefs, x)

    def fit(self, pts, order=3):
        xs, ys = pts[:,0], pts[:,1]
        self.coefs = np.polyfit(xs, ys, order)
        self.x_min, self.x_max = np.min(xs), np.max(xs)
    
    def draw_on_cam_img(self, img, cam, l0=0.1, l1=0.7, color=(0,128,0)):
        xs = np.linspace(l0, l1, 20); ys = self.get_y(xs)
        pts_world = np.array([[x, y, 0] for x, y in zip(xs, ys)])
        pts_img = cam.project(pts_world)
        for i in range(len(pts_img)-1):
            try:
                cv2.line(img, tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)), color, 4)
            except OverflowError:
                pass

            
class FakeLineDetector:
    # compute a fake detected line from path and localization
    def __init__(self):
        tdg_dir = rospkg.RosPack().get_path('two_d_guidance')
        default_path_filename = os.path.join(tdg_dir, 'paths/demo_z/track_trr_real.npz')
        path_filename = rospy.get_param('~path_filename', default_path_filename)

        rospy.loginfo(' loading path: {}'.format(path_filename))
        self.path = tdg.path.Path(load=path_filename)

        self.robot_pose_topic = rospy.get_param('~robot_pose_topic', "/nono_0/ekf/pose")
        self.robot_listener = ros_utils.GazeboTruthListener(topic=self.robot_pose_topic)
        rospy.loginfo(' getting robot pose from: {}'.format(self.robot_pose_topic))   

        self.path_body = None

        
    def compute_line(self):
        try:
            p0, psi = self.robot_listener.get_loc_and_yaw()
            p1, p2, end_reached, ip1, ip2 = self.path.find_carrot_looped(p0, _d=0.5)
            cy, sy = math.cos(psi), math.sin(psi)
            w2b = np.array([[cy, sy],[-sy, cy]])
            self.path_body = np.array([np.dot(w2b, p-p0) for p in self.path.points[ip1:ip2]])
            #print len(self.path_body)
            #if len(self.path_body) > 0:
            #    self.coefs = np.polyfit(self.path_body[:,0], self.path_body[:,1], 3)
            #pdb.set_trace()
            #print self.coefs
            
        except ros_utils.RobotNotLocalizedException:
            rospy.loginfo_throttle(1., "Robot not localized") # print every second
        except ros_utils.RobotLostException:
             rospy.loginfo_throttle(0.5, 'robot lost')

