#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import visualization_msgs.msg, geometry_msgs.msg
import math, numpy as np
import cv2

import two_d_guidance as tdg
import two_d_guidance.ros_utils as ros_utils
import two_d_guidance.msg


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
        self.coefs = np.polyfit(pts[:,0], pts[:,1], order)
        self.x_min, self.x_max = np.min(pts[:,0]), np.max(pts[:,0])
    
    def draw_on_cam_img(self, img, cam, l0=0.1, l1=0.7):
        xs = np.linspace(l0, l1, 20); ys = self.get_y(xs)
        pts_world = np.array([[x, y, 0] for x, y in zip(xs, ys)])
        pts_img = cam.project(pts_world)
        for i in range(len(pts_img)-1):
            try:
                cv2.line(img, tuple(pts_img[i].squeeze().astype(int)), tuple(pts_img[i+1].squeeze().astype(int)), (0,128,0), 4)
            except OverflowError:
                pass

class LaneModelPublisher:
    def __init__(self, topic='follow_line/detected_lane_model'):
        rospy.loginfo(' publishing lane model on ({})'.format(topic))
        self.pub = rospy.Publisher(topic, two_d_guidance.msg.LaneModel, queue_size=1)

    def publish(self, lm):
        msg = two_d_guidance.msg.LaneModel()
        msg.poly = lm.coefs
        self.pub.publish(msg)

        
class LaneModelSubscriber:
    def __init__(self, topic='follow_line/detected_lane_model'):
        self.sub = rospy.Subscriber(topic, two_d_guidance.msg.LaneModel, self.msg_callback, queue_size=1)
        rospy.loginfo(' subscribed to ({})'.format(topic))
        self.msg = None
        self.timeout = 0.5
        
    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()

    def get(self, lm):
        if self.msg is not None and (rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout:
            lm.coefs = self.msg.poly
            lm.set_valid(True)
        else:
            lm.set_valid(False)

class LaneModelMarkerPublisher:
    def __init__(self, ref_frame="nono_0/base_link_footprint", topic='/follow_line/detected_lane',
                 color=(1., 0., 1., 0.)):
        self.pub_lane = rospy.Publisher(topic, visualization_msgs.msg.Marker, queue_size=1)
        rospy.loginfo(' publishing lane model markers on ({})'.format(topic))
        self.lane_msg = visualization_msgs.msg.Marker()
        self.lane_msg.header.frame_id = ref_frame
        self.lane_msg.type = self.lane_msg.LINE_STRIP
        self.lane_msg.action = self.lane_msg.ADD
        self.lane_msg.id = 0
        self.lane_msg.text = 'lane'
        s = self.lane_msg.scale; s.x, s.y, s.z = 0.01, 0.2, 0.2
        c = self.lane_msg.color; c.a, c.r, c.g, c.b = color
        o = self.lane_msg.pose.orientation; o.x, o.y, o.z, o.w = 0, 0, 0, 1

    def publish(self, lm, l0=0.6, l1=1.8):
        #pts = [[0, 0, 0], [0.1, 0, 0], [0.2, 0.1, 0]]
        try:
            l0, l1 = lm.x_min, lm.x_max
        except AttributeError: pass
        pts = [[x, lm.get_y(x), 0] for x in np.linspace(l0, l1, 10)]
        self.lane_msg.points = []
        for p in pts:
            _p = geometry_msgs.msg.Point()
            _p.x, _p.y, _p.z = p
            self.lane_msg.points.append(_p)
        self.pub_lane.publish( self.lane_msg)  

            
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

