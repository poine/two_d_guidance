#!/usr/bin/env python

import sys, numpy as np, rospy, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg
import cv2, cv_bridge

import pdb

#import smocap
import smocap.rospy_utils
import follow_line, fl_utils as flu

class ImgPublisher:
    def __init__(self, cam_sys):
        img_topic = "/follow_line/image_debug"
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
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


class ContourPublisher:
    def __init__(self, frame_id='nono_0/base_link_footprint', topic='/follow_line/detected_contour'):
        self.frame_id = frame_id
        self.pub = rospy.Publisher(topic , visualization_msgs.msg.MarkerArray, queue_size=1)
        rospy.loginfo(' publishing contour on ({})'.format(topic))
        self.msg = visualization_msgs.msg.MarkerArray()

    def publish(self, contour_blf):
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = self.frame_id
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.id = 0
        marker.text = 'foo'
        marker.scale.x = 0.01
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        for x, y, z in contour_blf:
                p1 = geometry_msgs.msg.Point()
                p1.x=x; p1.y=y;p1.z=z
                marker.points.append(p1)
        self.msg.markers=[marker]
        self.pub.publish(self.msg)
        
        

'''

'''
class Node:

    def __init__(self):
        self.low_freq = 15.
        cam_names = rospy.get_param('~cameras', 'nono_0/camera1').split(',')
        ref_frame = rospy.get_param('~ref_frame', 'nono_0/base_link_footprint')
        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names, fetch_extrinsics=True, world=ref_frame)
        self.lane_finder = follow_line.LaneFinder()
        self.img_pub = ImgPublisher(self.cam_sys)
        self.cont_pub = ContourPublisher(frame_id=ref_frame)
        self.lane_model_marker_pub = flu.LaneModelMarkerPublisher(ref_frame=ref_frame)
        self.lane_model_pub = flu.LaneModelPublisher()
        self.lane_model = flu.LaneModel()
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)

    def on_image(self, img, (cam_idx, stamp, seq)):
        #pdb.set_trace()
        self.lane_finder.process_rgb_image(img, self.cam_sys.cameras[cam_idx])
        if self.lane_finder.floor_plane_injector.contour_floor_plane_blf is not None:
            self.lane_model.fit(self.lane_finder.floor_plane_injector.contour_floor_plane_blf[:,:2])
            self.lane_model_pub.publish(self.lane_model)
            
    def publish_image(self):
        self.img_pub.publish(self.cam_lst.get_images_as_rgb(), self.lane_finder)

    def publish_lane(self):
        if self.lane_finder.floor_plane_injector.contour_floor_plane_blf is not None:
            self.cont_pub.publish(self.lane_finder.floor_plane_injector.contour_floor_plane_blf)
        
    def periodic(self):
        self.publish_image()
        self.publish_lane()
        self.lane_model_marker_pub.publish(self.lane_model)
    
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass
        
def main(args):
  rospy.init_node('fl_lane_detect_node')
  rospy.loginfo('fl_lane_detect_node starting')
  rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
