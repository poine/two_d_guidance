#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import visualization_msgs.msg, geometry_msgs.msg
import math, numpy as np
import cv2
import pdb

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
# and landmarks
#
class TrrPath(tdg.path.Path):
    LM_START, LM_FINISH, LM_NB = range(3)
    lm_names = ['start', 'finish']
    def __init__(self, path_filename, v=0.6):
        tdg.path.Path.__init__(self)
        data = self.load(path_filename)
        try:
            self.vels = data['vels']
            self.accels = data['accels']
            self.jerks = data['jerks']
            self.offsets = data['offsets']
        except KeyError:
            print(' -no vel/acc/jerk/offsets in archive, setting them to zero')
            self.vels   = v*np.ones(len(self.points))
            self.accels = np.zeros(len(self.points))
            self.jerks  = np.zeros(len(self.points))
            self.offsets= np.zeros(len(self.points))

        try:
            self.track_points = data['track_points']
            self.speed_points = data['speed_points']
            self.offset_points = data['offset_points']
        except KeyError:
            print(' -no track/speed/offset definition, using path as default')
            self.track_points = np.array(data['points'])
            self.speed_points = [[0.0, 2.0]]
            self.offset_points = []
            
        try:
            self.lm_s = data['lm_s']  # landmark abscisses
        except KeyError:
            print(' -no landmark abscisses in path, using defaults')
            self.lm_s = [0., 16.66] # vedrines
        self.lm_idx, self.lm_points = [], []
        for _lm_s in self.lm_s:
            _i, _p = self.find_point_at_dist_from_idx(0, _d=_lm_s)
            self.lm_idx.append(_i) 
            self.lm_points.append(_p) 
            
        self.len = self.dists[-1] - self.dists[0]
        self.compute_time()
            
    def save(self, filename):
        print('saving path to {}'.format(filename))
        np.savez(filename, points=self.points, offsets=self.offsets, headings=self.headings, curvatures=self.curvatures, dists=self.dists, vels=self.vels, accels=self.accels, track_points=self.track_points, speed_points=self.speed_points, offset_points=self.offset_points, jerks=self.jerks, lm_s=self.lm_s)

    def build_path_only(self):
        self.points = np.array(self.track_points)
        self.compute_headings()
        self.offsets = np.zeros(len(self.points))
        for offset_position, offset_size, offset_length in self.offset_points:
            first_index = int((offset_position - offset_length / 2) * 100)
            length = int(offset_length * 100)
            if first_index + length >= len(self.points):
                first_index -= len(self.points)
            for i in range(length):
                offset = offset_size * (math.cos(math.pi * i * 2 / length - math.pi) + 1) * 0.5
                self.offsets[first_index + i] += offset
                self.points[first_index + i, 0] -= math.sin(self.headings[first_index + i]) * offset
                self.points[first_index + i, 1] += math.cos(self.headings[first_index + i]) * offset
                
    def compute_vels_1(self):
        '''
        Result is quite acceptable with simple configurations.
        CPU time efficient
        '''
        speed_points = sorted(self.speed_points, reverse=True)
        self.vels = np.zeros(len(self.points))
        index = len(self.vels)
        for position, value in speed_points: # Speed is set from the end to the start
            position = int(position * 100)
            self.vels[position:index] = value
            index = position
        forward_vels = self.vels
        forward_vels = filter(forward_vels, self.dists)[:,0]
        reverse_vels = np.flip(self.vels, 0)
        reverse_vels = filter(reverse_vels, np.flip(self.dists, 0) * -1)[:,0]
        self.vels = np.minimum(self.vels, np.minimum(forward_vels, np.flip(reverse_vels, 0)))
        #self.vels = filter(self.vels, self.dists)[:,0]

    

    def _update_dynamics(self, i, max_vel, accel_limits, jerk_limits, brake):
        '''
        Compute jerk, accel, vel for this point according to previous point
        Note: if previous vel/accel is not adequate, final val may exceed max_vel
        '''
        old_v = self.vels[i -1]
        dt = (self.dists[i] - self.dists[i - 1]) / old_v
        if dt == 0:
            self.jerks[i] = self.jerks[i - 1]
            self.accels[i] = self.accels[i - 1]
            self.vels[i] = self.vels[i - 1]
        else:
            old_a = self.accels[i - 1]
            if brake:
                j = jerk_limits[0]
            else:
                j = jerk_limits[1]
            a = old_a + j * dt
            if a > accel_limits[1]:
                a = accel_limits[1]
                j = (a - old_a) / dt
            if a < accel_limits[0]:
                a = accel_limits[0]
                j = (a - old_a) / dt
            v = old_v + a * dt
            if v > max_vel and not brake:
                a = (max_vel - old_v) / dt
                j = (a - old_a) / dt
                if a >= accel_limits[0] and j >= jerk_limits[0]:
                    v = max_vel
                    
            self.jerks[i] = j
            self.accels[i] = a
            self.vels[i] = v

    def _braking_reverse_time(self, i, max_vel, accel_limits, jerk_limits):
        '''
        Compute vel, and accel for point i based on next point.
        Next point accel is sometimes adjusted
        All equations are reversed to get the best slow down to
        the next point.
        '''
        dd = self.dists[i + 1] - self.dists[i]
        if dd == 0.:
            self.jerks[i] = self.jerks[i + 1]
            self.accels[i] = self.accels[i + 1]
            self.vels[i] = self.vels[i + 1]
        else:
            if self.vels[i + 1] > max_vel:
                self.vels[i] = max_vel
                dt = (self.dists[i + 1] - self.dists[i]) / self.vels[i]
                self.accels[i] = -jerk_limits[1] * dt
            else:
                delta = self.vels[i + 1] * self.vels[i + 1] - 4 * self.accels[i + 1] * dd
                self.vels[i] = (self.vels[i + 1] + math.sqrt(delta)) / 2
                if self.vels[i] > max_vel:
                    self.vels[i] = max_vel
                    dt = dd / self.vels[i]
                    self.accels[i + 1] = (self.vels[i + 1] - self.vels[i]) / dt
                else:
                    dt = dd / self.vels[i]
                self.accels[i] = self.accels[i + 1] - self.jerks[i + 1] * dt
                if self.accels[i] < accel_limits[0]:
                    self.accels[i] = accel_limits[0]
            self.jerks[i] = jerk_limits[1]
            
    def compute_vels(self):
        '''
        Strict respect of max speed, accel and jerk limits
        CPU time consmming
        Not perfectly smooth
        '''
        accel_limits = [-4, 6]
        jerk_limits = [-15, 25]
        speed_points = sorted(self.speed_points, reverse=True)
        self.accels = np.zeros(len(self.points))
        self.jerks = np.zeros(len(self.points))
        self.vels = np.zeros(len(self.points))
        max_vels = np.zeros(len(self.points))
        index = len(max_vels)
        for position, value in speed_points: # Speed is set from the end to the start
            position = int(position * 100)
            max_vels[position:index] = value
            index = position
        
        # Compute end of brakings from the end
        self.jerks[-1] = jerk_limits[1]
        self.vels[-1] = max_vels[-1]
        self.vels[0] = max_vels[0]
        dt = np.linalg.norm(self.points[0] - self.points[-1]) / self.vels[-1]
        self.accels[-1] = -jerk_limits[1] * dt
        i = len(max_vels) - 2
        while i >= 0:
            self._braking_reverse_time(i, max_vels[i], accel_limits, jerk_limits)
            max_vels[i] = self.vels[i]
            i -= 1

        i = 0
        slow_down_point = None
        constraint_position = None
        while i < len(max_vels):
            brake = constraint_position is not None and i < constraint_position
            self._update_dynamics(i, max_vels[i], accel_limits, jerk_limits, brake)
            if constraint_position is not None and i == constraint_position:
                if self.vels[i] <= max_vels[i] + 0.00001:
                    slow_down_point = None
                    constraint_position = None
                else:
                    slow_down_point -= 1
                    i = slow_down_point - 1
                    
            if constraint_position is None and self.vels[i] > max_vels[i]: # + 0.0001:
                slow_down_point = i - 1
                constraint_position = i
                i = slow_down_point - 1

            i += 1
                    
                    
    def compute_all(self):
        self.build_path_only()
        self.compute_headings()
        self.compute_curvatures()
        self.compute_dists()
        self.len = self.dists[-1] - self.dists[0]
        self.compute_vels()
        self.compute_time()
        self.accels = np.zeros(len(self.points))
        d0 = np.linalg.norm(self.points[0] - self.points[-1])
        if d0 != 0.:
            self.accels[0] = (self.vels[0] - self.vels[-1]) / (d0 / self.vels[-1])
        for i in range(1, len(self.vels)):
            dt = self.time[i] - self.time[i - 1]
            if dt == 0.:
                self.accels[i] = self.accels[i - 1]
            else:
                self.accels[i] = (self.vels[i] - self.vels[i - 1]) / dt

        self.jerks = np.zeros(len(self.points))
        if d0 != 0:
            self.jerks[0] = (self.accels[0] - self.accels[-1]) / (d0 / self.vels[-1])
        for i in range(1, len(self.accels)):
            dt = self.time[i] - self.time[i - 1]
            if dt == 0.:
                self.jerks[i] = self.jerks[i - 1]
            else:
                self.jerks[i] = (self.accels[i] - self.accels[i - 1]) / (self.time[i] - self.time[i - 1])


    def compute_time(self):
        # integrate
        self.time = np.zeros(len(self.points))
        for i in range(1, len(self.time)):
            d = self.dists[i]-self.dists[i-1]
            self.time[i] = self.time[i-1] + d/self.vels[i-1]

            
    def report(self):
        rospy.loginfo(' lap time: {:.2f} s'.format(self.time[-1]))
        rospy.loginfo(' path len {:.2f} m, {} points'.format(self.len, len(self.dists)))
        rospy.loginfo(' path control points {} speed, {} deviation'.format(len(self.speed_points), len(self.offset_points)))
        rospy.loginfo(' path start {} (dist {:.2f}) '.format(self.points[0], self.dists[0]))
        rospy.loginfo(' path finish {}(dist {:.2f})'.format(self.points[-1], self.dists[-1]))
        rospy.loginfo('  lm_start idx {} pos {} dist {:.2f}'.format(self.lm_idx[self.LM_START], self.lm_points[self.LM_START], self.lm_s[self.LM_START]))
        rospy.loginfo('  lm_finish idx {} pos {} dist {:.2f}'.format(self.lm_idx[self.LM_FINISH], self.lm_points[self.LM_FINISH], self.lm_s[self.LM_FINISH]))
        rospy.loginfo('  vels min/avg/max {:.1f} {:.1f} {:.1f} m/s'.format(np.min(self.vels), self.dists[-1] / self.time[-1], np.max(self.vels)))
        rospy.loginfo('  accels min/avg/max {:.1f} {:.1f} {:.1f} m/s^2'.format(np.min(self.accels), np.mean(self.accels), np.max(self.accels)))

        
#
# Velocity profile
#
# Same values as used in guidance
def filter(vel_sps, dists, omega=6., xi=0.9):
    # second order reference model driven by input setpoint
    _sats = [4., 25.]  # accel, jerk
    ref = tdg.utils.SecOrdLinRef(omega=omega, xi=xi, sats=_sats)
    out = np.zeros((len(vel_sps), 3))
    # run ref
    out[0, 0] = vel_sps[0]; ref.reset(out[0])
    for i in range(1,len(vel_sps)):
        dt = (dists[i] - dists[i - 1]) / out[i - 1, 0]
        out[i] = ref.run(dt, vel_sps[i])
    return out


import matplotlib.pyplot as plt
class LaneModel:
    # center line as polynomial
    # y = an.x^n + ...
    def __init__(self):
        self.order = 3
        self.coefs = [0., 0., 0., 0.01, 0.05]
        self.stamp = None
        self.valid = False
        self.inliers_mask = []

    def is_valid(self): return self.valid
    def set_valid(self, v): self.valid = v

    def set_invalid(self):
        self.valid = False
        self.coefs = np.full(self.order+1, np.nan)
  
    def get_y(self, x):
        return np.polyval(self.coefs, x)

    def fit_all_contours(self, ctrs, order=3):
        xs, ys, weights = [], [], []
        for c in ctrs:
            area = cv2.contourArea(c)
            xs = np.append(xs, c[:, 0, 0])
            ys = np.append(ys, c[:, 0, 1])
            weights = np.append(weights, [area / len(c)] * len(c))

        self.coefs, _res, rank, _singular, _rcond = np.polyfit(xs, ys, order, full=True, w=weights)
        if rank <= order:
            reduced_order = min(1, rank - 2)
            self.coefs = np.polyfit(xs, ys, reduced_order, w=weights)
            self.coefs = np.append([0] * (order - reduced_order), self.coefs)
        self.x_min, self.x_max = np.min(xs), np.max(xs)

    def fit(self, ctrs, order=3, right_border=0.9, left_border=-0.9, lateral_margin=0.25):
        """ 
        Input: ctrs are contours with points coordinates in meters.
        ctrs[index][:, 0, 0=front dist/1=lateral offset]
        Origin is robot center
        """
        self.inliers_mask = np.full(len(ctrs), True)
        if len(ctrs) < 2:
            self.fit_all_contours(ctrs, order)
        else:
            all_min = [min(c[:, 0, 0]) for c in ctrs]
            all_max = [max(c[:, 0, 0]) for c in ctrs]
            # low ref at 1/4 of the contour, high ref at 3/4 of the contour
            low_ref_point = np.multiply(np.add(np.multiply(all_min, 3), all_max), 0.25)
            high_ref_point = np.multiply(np.add(all_min, np.multiply(all_max, 3)), 0.25)

            # priority is defined by (max - min) / min = max/min - 1, but -1 constant is simplified
            # offset to increase priority of near contours
            offset = min(all_min) / 2
            all_priorities = np.divide(np.add(all_max, -offset), np.add(all_min, -offset))
            
            remaining = len(ctrs)
            selected = [ ]
            while True:
                best_id = np.argmax(all_priorities)
                candidate = ctrs[best_id]
                this_min = all_min[best_id]
                this_max = all_max[best_id]
                all_priorities[best_id] = -1
                remaining -= 1

                # New contour has to match with building curve
                if len(selected) > 0:
                    self.fit_all_contours(selected, order)
                    y1 = np.polyval(self.coefs, this_min)
                    y2 = np.polyval(self.coefs, this_max)
                    index1 = np.argmin(candidate[:, 0, 0])
                    d1 = candidate[index1, 0, 1] - y1
                    index2 = np.argmax(candidate[:, 0, 0])
                    d2 = candidate[index2, 0, 1] - y2
                    if min(abs(d1), abs(d2)) > lateral_margin:
                        self.inliers_mask[best_id] = False
                        if remaining == 0:
                            break
                        else:
                            continue
                        
                selected.append(candidate)

                # If the new contour reaches a border, then stop
                if max(candidate[:, 0, 1]) > right_border or min(candidate[:, 0, 1]) < left_border :
                    for i in range(len(all_priorities)):
                        if all_priorities[i] > 0:
                            self.inliers_mask[i] = False
                    break

                # Invalidate overlapping contours
                for i in range(len(all_priorities)):
                    if all_priorities[i] > 0:
                        if low_ref_point[i] < this_max and high_ref_point[i] > this_min :
                            all_priorities[i] = -1
                            self.inliers_mask[i] = False
                            remaining -= 1
                            
                if remaining == 0:
                    break

            self.fit_all_contours(selected, order)

    def _plot(self, ctrs):
        for c in ctrs:
            plt.plot(c[:,0,0], c[:,0,1])
        xs = np.linspace(self.x_min, self.x_max)
        plt.plot(xs, np.polyval(self.coefs,xs))
        #pdb.set_trace()
        #plt.gca().axis('equal')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

            
            
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

