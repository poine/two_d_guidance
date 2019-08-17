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
        
class ImgPublisher:
    def __init__(self, img_topic = "/trr_guidance/image_debug"):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic+"/compressed", sensor_msgs.msg.CompressedImage, queue_size=1)
        cam_names = ['/camera_road_front'] #['caroline/camera_road_front']
        self.img = None
        #self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)
        self.img_sub = rospy.Subscriber("/camera_road_front/image_raw/compressed", sensor_msgs.msg.CompressedImage, self.img_cbk,  queue_size = 1)
        self.compressed_img = None
        
    def img_cbk(self, msg):
        self.compressed_img = np.fromstring(msg.data, np.uint8)
        
        
    def on_image(self, img, (cam_idx, stamp, seq)):
        # store subscribed image (bgr, opencv)
        self.img = img
    
    def publish(self, mode, lin_sp, lin_odom, ang_sp, ang_odom, lane_model, lookahead):
        if self.compressed_img is not None:
            self.img = cv2.cvtColor(cv2.imdecode(self.compressed_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            self.draw(mode, lin_sp, lin_odom, ang_sp, ang_odom, lane_model, lookahead)
            img_rgb = self.img[...,::-1] # rgb = bgr[...,::-1] OpenCV image to Matplotlib
            msg = sensor_msgs.msg.CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img_rgb)[1]).tostring()
            self.image_pub.publish(msg)

    def draw(self, mode, lin_sp, lin_odom, ang_sp, ang_odom, lane_model, lookahead):
        f, h, c, w = cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2
        str_of_guid_mode = ['idle', 'stopped', 'driving']
        cv2.putText(self.img, 'mode: {:s} curv: {:-6.2f}'.format(str_of_guid_mode[mode], lane_model.coefs[1]), (20, 40), f, h, c, w)
        lookahead_time = np.inf if lin_sp == 0 else  lookahead/lin_sp
        cv2.putText(self.img, 'lookahead: {:.2f}m {:.2f}s'.format(lookahead, lookahead_time), (20, 90), f, h, c, w)
        cv2.putText(self.img, 'lin:  sp/odom {:.2f}/{:.2f} m/s'.format(lin_sp, lin_odom), (20, 140), f, h, c, w)
        cv2.putText(self.img, 'ang: sp/odom {: 6.2f}/{: 6.2f} deg/s'.format(np.rad2deg(ang_sp), np.rad2deg(ang_odom)), (20, 190), f, h, c, w)



class GuidanceStatusSubscriber:
    def __init__(self, topic='trr_guidance/status'):
        self.sub = rospy.Subscriber(topic, two_d_guidance.msg.FLGuidanceStatus, self.msg_callback, queue_size=1)
        rospy.loginfo(' subscribed to ({})'.format(topic))
        self.msg = None
        self.timeout = 0.5
        
    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()

    def get(self, lm):
        if self.msg is not None and (rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout:
            pass
            #lm.coefs = self.msg.poly
            #lm.set_valid(True)
        else:
            #lm.set_valid(False)
            pass



class Node:

    def __init__(self):
        self.freq = 10
        ref_frame = rospy.get_param('~ref_frame', 'nono_0/base_link_footprint')
        rospy.loginfo(' using ref_frame: {}'.format(ref_frame))
        self.mark_pub = MarkerPublisher(ref_frame)
        self.img_pub = ImgPublisher()
        self.guid_stat_sub = GuidanceStatusSubscriber()
        self.lane_model = trr_u.LaneModel()
        
    def periodic(self):
        m = self.guid_stat_sub.msg
        if m is not None:
            R, carrot = m.R, [m.carrot_x, m.carrot_y]
            self.mark_pub.publish_carrot(carrot)
            self.mark_pub.publish_arc(R, carrot)
            self.lane_model.coefs = m.poly
            self.mark_pub.publish_lane(self.lane_model)
            #print self.guid_stat_sub.msg
            self.img_pub.publish(m.guidance_mode, m.lin_sp, 0., m.ang_sp, 0, self.lane_model, m.lookahead_dist)

    def run(self):
        rate = rospy.Rate(self.freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

def main(args):
  rospy.init_node('trr_guidance_display_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
