#!/usr/bin/env python

import sys, numpy as np, rospy, sensor_msgs.msg
import cv2, cv_bridge

import pdb

#import smocap
import smocap.rospy_utils
import follow_line

class ImgPublisher:
    def __init__(self, cam_sys):
        img_topic = "/follow_line/image_debug"
        rospy.loginfo(' publishing on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.cam_sys = cam_sys
        w, h = np.sum([cam.w for cam in  cam_sys.get_cameras()]), np.max([cam.h for cam in cam_sys.get_cameras()])
        self.img = 255*np.ones((h, w, 3), dtype='uint8')
        
    def publish(self, imgs, lf):
        x0 = 0
        for cam, img in zip(self.cam_sys.get_cameras(), imgs):
            if  img is not None:
                h, w = img.shape[0], img.shape[1]; x1 = x0+w
                self.img[:h,x0:x1] = lf.draw(img, cam)
                x0 = x1
        if 0:
            x0 = 0
            for i, (img, cam) in enumerate(zip(imgs, self.cam_sys.get_cameras())):
                if img is not None:
                    h, w = img.shape[0], img.shape[1]; x1 = x0+w
                    cam_area = self.img[:h,x0:x1]
                    self.img[:h,x0:x1] = img
                    cv2.rectangle(cam_area, (0, 0), (x1-1, h-1), (0,255,0), 2)
                    x0 = x1

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "rgb8"))

'''

'''
class Node:

    def __init__(self):
        self.low_freq = 15.
        cam_names = rospy.get_param('~cameras', 'oscar_v0/camera1').split(',')
        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names, fetch_extrinsics=False)
        self.lane_finder = follow_line.LaneFinder()
        self.img_pub = ImgPublisher(self.cam_sys)
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)

    def on_image(self, img, (cam_idx, stamp, seq)):
        #pdb.set_trace()
        self.lane_finder.process_image(img, cam_idx)

    def periodic(self):
        self.img_pub.publish(self.cam_lst.get_images_as_rgb(), self.lane_finder)

    
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
  rospy.loginfo('follow_line_node starting')
  rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
