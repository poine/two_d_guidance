#!/usr/bin/env python
import os, sys
import math, numpy as np
import roslib, rospy, rospkg, rostopic, dynamic_reconfigure.server
import nav_msgs.msg , geometry_msgs.msg, visualization_msgs.msg, sensor_msgs.msg

import two_d_guidance.msg
import two_d_guidance.trr.utils as trr_u
import two_d_guidance.trr.rospy_utils as trr_rpu

import smocap.rospy_utils, cv2


class MarkerPublisher:
    def __init__(self, ref_frame):
        self.pub_carrot = rospy.Publisher('trr_guidance/goal', visualization_msgs.msg.Marker, queue_size=1)
        self.carrot_msg = visualization_msgs.msg.Marker()
        self.carrot_msg.header.frame_id=ref_frame
        self.carrot_msg.type = visualization_msgs.msg.Marker.CYLINDER
        p = self.carrot_msg.pose.position; p.x, p.y, p.z = 0, 0, 0.025
        o = self.carrot_msg.pose.orientation; o.x, o.y, o.z, o.w = 0, 0, 0, 1
        s = self.carrot_msg.scale; s.x, s.y, s.z = 0.01, 0.01, 0.1
        c = self.carrot_msg.color; c.a, c.r, c.g, c.b = 1., 0., 1., 0.

        self.pub_arc = rospy.Publisher('trr_guidance/arc', nav_msgs.msg.Path, queue_size=1)
        self.arc_msg = nav_msgs.msg.Path()
        self.arc_msg.header.frame_id = ref_frame

        self.pub_lane = trr_rpu.LaneModelMarkerPublisher(ref_frame=ref_frame, topic='trr_guidance/detected_lane_model_guidance', color=(1, 1, 0, 0))
        
        #self.img_pub = ImgPublisher()
        
    def publish_carrot(self, carrot_pos):
        self.carrot_msg.header.stamp = rospy.Time.now()
        p = self.carrot_msg.pose.position; p.x, p.y, p.z = carrot_pos[0], carrot_pos[1], 0
        self.pub_carrot.publish(self.carrot_msg)

    def publish_arc(self, R, carrot_pos):
        self.arc_msg.header.stamp = rospy.Time.now()
        self.arc_msg.poses = []
        alpha = np.arctan(carrot_pos[0]/(R - carrot_pos[1]))
        for theta in np.linspace(0, alpha, 20):
            pose = geometry_msgs.msg.PoseStamped()
            pose.pose.position.x =  R*np.sin(theta)   if not math.isinf(R) else 0
            pose.pose.position.y = -R*np.cos(theta)+R if not math.isinf(R) else 0
            self.arc_msg.poses.append(pose)
        self.pub_arc.publish(self.arc_msg)

    def publish_lane(self, lm):
        self.pub_lane.publish(lm)
        
class ImgPublisher(trr_rpu.DebugImgPublisher):
    def __init__(self, img_topic, cam_name):
        trr_rpu.DebugImgPublisher.__init__(self, cam_name, img_topic)
 
    def _draw(self, img_bgr, model, data):
        y0=20; font_color=(128,0,255)
        f, h1, h2, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1.25, 0.9, font_color, 2
        cv2.putText(img_bgr, 'Guidance:', (y0, 40), f, h1, c, w)

        msg = model.guid_stat_sub.get()
        str_of_guid_mode = ['idle', 'stopped', 'driving']
        mode_name, curv = str_of_guid_mode[msg.guidance_mode], msg.poly[1]
        cv2.putText(img_bgr, 'mode: {:s} curv: {:-6.2f}'.format(mode_name, curv), (20, 90), f, h2, c, w)
        
        lin_odom, ang_odom = model.odom_sub.get_vel()
        lin_sp, ang_sp = msg.lin_sp, msg.ang_sp
        lookahead = msg.lookahead_dist
        lookahead_time = np.inf if lin_sp == 0 else  lookahead/lin_sp
        cv2.putText(img_bgr, 'lookahead: {:.2f}m {:.2f}s'.format(lookahead, lookahead_time), (20, 140), f, h2, c, w)
        cv2.putText(img_bgr, 'lin:  sp/odom {:.2f}/{:.2f} m/s'.format(lin_sp, lin_odom), (20, 190), f, h2, c, w)
        cv2.putText(img_bgr, 'ang: sp/odom {: 6.2f}/{: 6.2f} deg/s'.format(np.rad2deg(ang_sp), np.rad2deg(ang_odom)), (20, 240), f, h2, c, w)






class Node(trr_rpu.PeriodicNode):

    def __init__(self):
        trr_rpu.PeriodicNode.__init__(self, 'trr_guidance_display_node')
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        cam_name = rospy.get_param('~camera', prefix(robot_name, 'camera_road_front'))
        ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))
        rospy.loginfo(' using ref_frame: {}'.format(ref_frame))
        self.lane_model = trr_u.LaneModel()
        self.mark_pub = MarkerPublisher(ref_frame)
        self.img_pub = ImgPublisher("/trr_guidance/image_debug", cam_name)
        self.guid_stat_sub = trr_rpu.GuidanceStatusSubscriber()
        self.odom_sub = trr_rpu.OdomListener('/odom', 'guidance_display_node')
        
    def periodic(self):
        try:
            m = self.guid_stat_sub.get()
            R, carrot = m.R, [m.carrot_x, m.carrot_y]
            self.mark_pub.publish_carrot(carrot)
            self.mark_pub.publish_arc(R, carrot)
            self.lane_model.coefs = m.poly
            self.mark_pub.publish_lane(self.lane_model)
        except trr_rpu.NoRXMsgException :
            print('guidance display: no Status received from guidance')
        except trr_rpu.RXMsgTimeoutException :
            print('guidance display: timeout receiving Status from guidance')

        try:
            self.img_pub.publish(self, None)
        except trr_rpu.NoRXMsgException :
            print('guidance display im: no Status received from guidance')
        except trr_rpu.RXMsgTimeoutException :
            print('guidance display im: timeout receiving Status from guidance')

            

def main(args):
  rospy.init_node('trr_guidance_display_node')
  Node().run(10)


if __name__ == '__main__':
    main(sys.argv)
