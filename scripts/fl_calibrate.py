#!/usr/bin/env python

import sys, numpy as np, rospy, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg
import cv2, cv_bridge

import smocap, smocap.rospy_utils


class Node:

    def __init__(self):
        self.low_freq = 15.
        cam_names = rospy.get_param('~cameras', 'nono_0/camera1').split(',')
        ref_frame = rospy.get_param('~ref_frame', 'nono_0/base_link_footprint')
        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names, fetch_extrinsics=True, world=ref_frame)
        img_topic = "/fl_calibrate/image"
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.img = None
        self.points_world = np.array([[1, -0.5, 0], [1, 0.5, 0], [2, 0.5, 0], [2, -0.5, 0]])
        self.points_img = self.cam_sys.cameras[0].project(self.points_world)

        self.fov_publisher = smocap.rospy_utils.FOVPublisher(self.cam_sys, frame_id=ref_frame)
        
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)

        
    def on_image(self, img, (cam_idx, stamp, seq)):
        self.img = img
    
    def periodic(self):
        #self.img_pub.publish(self.cam_lst.get_images_as_rgb(), self)
        if self.img is not None:
            #cv2.polylines(self.img, self.pts_world, isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.polylines(self.img, self.points_img.astype(np.int), isClosed=True, color=(255, 0, 0), thickness=2)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "rgb8"))
        self.fov_publisher.publish()
        
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass


def main(args):
  rospy.init_node('follow_line_node')
  rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
