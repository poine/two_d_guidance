#!/usr/bin/env python

import sys, numpy as np, rospy,  dynamic_reconfigure.server, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg
import cv2, cv_bridge

import pdb

#import smocap
import smocap.rospy_utils

import two_d_guidance.trr_utils as trru, two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_rospy_utils as trr_rpu
import two_d_guidance.trr.vision.lane_1 as trr_l1
import two_d_guidance.trr.vision.lane_2 as trr_l2

import two_d_guidance.cfg.trr_vision_laneConfig

# class ImgPublisher:
#     def __init__(self, cam_sys, img_topic):
#         self.publish_compressed = True
#         rospy.loginfo(' publishing image on ({})'.format(img_topic))
#         if self.publish_compressed:
#             self.image_pub = rospy.Publisher(img_topic+"/compressed", sensor_msgs.msg.CompressedImage, queue_size=1)
#         else:
#             self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
#             self.bridge = cv_bridge.CvBridge()
#         self.cam_sys = cam_sys
#         w, h = np.sum([cam.w for cam in  cam_sys.get_cameras()]), np.max([cam.h for cam in cam_sys.get_cameras()])
#         self.img = 255*np.ones((h, w, 3), dtype='uint8')
        
#     def publish(self, imgs, pipeline):
#         x0 = 0
#         for cam, img in zip(self.cam_sys.get_cameras(), imgs):
#             if  img is not None:
#                 h, w = img.shape[0], img.shape[1]; x1 = x0+w
#                 self.img[:h,x0:x1] = pipeline.draw_debug(cam, img)
#                 x0 = x1
#         if self.publish_compressed:
#             msg = sensor_msgs.msg.CompressedImage()
#             msg.header.stamp = rospy.Time.now()
#             msg.format = "jpeg"
#             msg.data = np.array(cv2.imencode('.jpg', self.img)[1]).tostring()
#             self.image_pub.publish(msg)
#         else:
#             self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "rgb8"))




class BirdEyePublisher(trr_rpu.ContourPublisher):
    def __init__(self, frame_id, be, topic='trr_vision/lane/bird_eye'):
        trr_rpu.ContourPublisher.__init__(self, frame_id, topic, be.param.va_bf, rgba=(1.,0.,0.,1.))
        

'''

'''
class Node(trr_rpu.TrrSimpleVisionPipeNode):

    def __init__(self):
        pipe_type = 1
        pipe_classes = [trr_l1.Contour1Pipeline, trr_l2.Contour2Pipeline, trr_vu.Foo3Pipeline]
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, pipe_classes[pipe_type], self.pipe_cbk)
        try:
            self.pipeline.bird_eye.set_param(self.cam, trr_vu.CarolineBirdEyeParam())
        except AttributeError: rospy.loginfo("  NoBE")
   
        # Publishing
        # Image
        self.img_pub = trr_rpu.CompressedImgPublisher(self.cam, '/trr_vision/lane/image_debug')
        # Markers
        self.cont_pub = trr_rpu.ContourPublisher(topic='/trr_vision/lane/detected_contour_markers', frame_id=self.ref_frame)
        self.fov_pub = smocap.rospy_utils.FOVPublisher(self.cam_sys, self.ref_frame, '/trr_vision/lane/fov')
        try:
            self.be_pub = BirdEyePublisher(self.ref_frame, self.pipeline.bird_eye, '/trr_vision/lane/bird_eye')
        except AttributeError:
            self.be_pub = None
        self.lane_model_marker_pub = trru.LaneModelMarkerPublisher('/trr_vision/lane/detected_model_markers', ref_frame=self.ref_frame)
        # Model
        self.lane_model_pub = trru.LaneModelPublisher('/trr_vision/lane/detected_model')
        self.lane_model = trru.LaneModel()
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_vision_laneConfig, self.cfg_callback)

    def cfg_callback(self, config, level):
        rospy.loginfo("  Reconfigure Request:")
        self.pipeline.thresholder.set_threshold(config['mask_threshold'])
        self.pipeline.display_mode = config['display_mode']
        return config

    def pipe_cbk(self):
        if self.pipeline.lane_model.is_valid():
            self.lane_model_pub.publish(self.pipeline.lane_model)
            
    def publish_3Dmarkers(self):
        #self.fov_pub.publish()
        if self.be_pub is not None:
            self.be_pub.publish()
            if self.pipeline.bird_eye.cnt_fp is not None:
                self.cont_pub.publish(self.pipeline.bird_eye.cnt_fp)
        self.lane_model_marker_pub.publish(self.pipeline.lane_model)

            
    def periodic(self):
        if  self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
            #self.publish_image()
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
    name = 'trr_vision_lane_node'
    rospy.init_node(name)
    rospy.loginfo('{} starting'.format(name))
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run()


if __name__ == '__main__':
    main(sys.argv)
