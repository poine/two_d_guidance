#!/usr/bin/env python

import sys, numpy as np, rospy,  dynamic_reconfigure.server, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg
import cv2, cv_bridge

import pdb

#import smocap
import smocap.rospy_utils

import two_d_guidance.trr_utils as trru, two_d_guidance.trr_vision_utils as trrvu
import two_d_guidance.trr_rospy_utils as trr_rpu
import two_d_guidance.trr.vision.lane_1 as trr_l1
import two_d_guidance.trr.vision.lane_2 as trr_l2

import two_d_guidance.cfg.fl_lane_detectorConfig

class ImgPublisher:
    def __init__(self, cam_sys, img_topic):
        self.publish_compressed = True
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        if self.publish_compressed:
            self.image_pub = rospy.Publisher(img_topic+"/compressed", sensor_msgs.msg.CompressedImage, queue_size=1)
        else:
            self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
            self.bridge = cv_bridge.CvBridge()
        self.cam_sys = cam_sys
        w, h = np.sum([cam.w for cam in  cam_sys.get_cameras()]), np.max([cam.h for cam in cam_sys.get_cameras()])
        self.img = 255*np.ones((h, w, 3), dtype='uint8')
        
    def publish(self, imgs, pipeline):
        x0 = 0
        for cam, img in zip(self.cam_sys.get_cameras(), imgs):
            if  img is not None:
                h, w = img.shape[0], img.shape[1]; x1 = x0+w
                self.img[:h,x0:x1] = pipeline.draw_debug(cam, img)
                x0 = x1
        if self.publish_compressed:
            msg = sensor_msgs.msg.CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', self.img)[1]).tostring()
            self.image_pub.publish(msg)
        else:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "rgb8"))




class BirdEyePublisher(trr_rpu.ContourPublisher):
    def __init__(self, frame_id, be, topic='follow_line/bird_eye'):
        trr_rpu.ContourPublisher.__init__(self, frame_id, topic, be.param.va_bf, rgba=(1.,0.,0.,1.))
        

'''

'''
class Node:

    def __init__(self):
        self.low_freq = 10.
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        cam_names = rospy.get_param('~cameras', prefix(robot_name, 'camera_road_front')).split(',')
        ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))
        # Camera System
        if 1:
            self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names, fetch_extrinsics=True, world=ref_frame)
        else:
            self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(cam_names, fetch_extrinsics=False, world=ref_frame)
            world_to_camo_t = [ 0.00039367,  0.3255541,  -0.04368782]
            world_to_camo_q = [ 0.6194402,  -0.61317113,  0.34761617,  0.34565589]
            self.cam_sys.cameras[0].set_location(world_to_camo_t, world_to_camo_q)
        self.cam_sys.cameras[0].set_undistortion_param(alpha=1.)
        # Pipeline
        pipe=2
        if pipe == 1: self.pipeline = trr_l1.Contour1Pipeline(self.cam_sys.cameras[0])
        elif pipe == 2:
            self.pipeline = trr_l2.Contour2Pipeline(self.cam_sys.cameras[0], trrvu.CarolineBirdEyeParam())
            self.pipeline.display_mode = trr_l2.Contour2Pipeline.show_be
        elif pipe == 3:
            self.pipeline = trrvu.Foo3Pipeline(self.cam_sys.cameras[0])
        # Publishing
        # Image
        self.img_pub = ImgPublisher(img_topic='/trr_vision/lane/image_debug', cam_sys=self.cam_sys)
        # Markers
        self.cont_pub = trr_rpu.ContourPublisher(topic='/trr_vision/lane/detected_contour_markers', frame_id=ref_frame)
        self.fov_pub = smocap.rospy_utils.FOVPublisher(self.cam_sys, ref_frame, '/trr_vision/lane/fov')
        try:
            self.be_pub = BirdEyePublisher(ref_frame, self.pipeline.bird_eye, '/trr_vision/lane/bird_eye')
        except AttributeError:
            self.be_pub = None
        self.lane_model_marker_pub = trru.LaneModelMarkerPublisher('/trr_vision/lane/detected_model_markers', ref_frame=ref_frame)
        # Model
        self.lane_model_pub = trru.LaneModelPublisher('/trr_vision/lane/detected_model')
        self.lane_model = trru.LaneModel()
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=cam_names, cbk=self.on_image)
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.fl_lane_detectorConfig, self.cfg_callback)

    def cfg_callback(self, config, level):
        rospy.loginfo("  Reconfigure Request:")
        #pdb.set_trace()
        print config, level
        #rospy.loginfo("  Reconfigure Request: {int_param}, {lookahead}, {str_param}, {bool_param}, {size}".format(**config))
        self.pipeline.thresholder.set_threshold(config['mask_threshold'])
        self.pipeline.display_mode = config['display_mode']
        return config

    
    def on_image(self, img, (cam_idx, stamp, seq)):
        #pdb.set_trace()
        #print(img.dtype)
        self.pipeline.process_image(img, self.cam_sys.cameras[cam_idx], stamp, seq)
        if self.pipeline.lane_model.is_valid():
            self.lane_model_pub.publish(self.pipeline.lane_model)
            
    def publish_image(self):
        self.img_pub.publish(self.cam_lst.get_images_as_rgb(), self.pipeline)

    def publish_3Dmarkers(self):
        #self.fov_pub.publish()
        if self.be_pub is not None:
            self.be_pub.publish()
            if self.pipeline.bird_eye.cnt_fp is not None:
                self.cont_pub.publish(self.pipeline.bird_eye.cnt_fp)
        self.lane_model_marker_pub.publish(self.pipeline.lane_model)

            
    def periodic(self):
        self.publish_image()
        self.publish_3Dmarkers()
    
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
