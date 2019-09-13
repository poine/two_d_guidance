#!/usr/bin/env python

import sys, numpy as np, rospy,  dynamic_reconfigure.server, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg
import cv2, cv_bridge

import pdb

#import smocap
import smocap.rospy_utils

import two_d_guidance.trr.utils as trru
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.rospy_utils as trr_rpu
# Different versions
import two_d_guidance.trr.vision.lane_1 as trr_l1 # Contour detection in thresholded gray camera image
import two_d_guidance.trr.vision.lane_2 as trr_l2 # Contour detection in thresholded gray camera image, fit in bird eye
import two_d_guidance.trr.vision.lane_3 as trr_l3 # Contour detection in thresholded bird eye image
import two_d_guidance.trr.vision.lane_4 as trr_l4 # In progress

import two_d_guidance.cfg.trr_vision_laneConfig

'''

'''
class Node(trr_rpu.TrrSimpleVisionPipeNode):

    def __init__(self):
        pipe_type = 2
        pipe_classes = [trr_l1.Contour1Pipeline, trr_l2.Contour2Pipeline, trr_l3.Contour3Pipeline, trr_l4.Foo4Pipeline ]
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, pipe_classes[pipe_type], self.pipe_cbk)
        
        roi_y_min = rospy.get_param('~roi_y_min', 0)
        tl, br = (0, roi_y_min), (self.cam.w, self.cam.h)
        self.pipeline.set_roi(tl, br) 
        # Image publishing
        self.img_pub = trr_rpu.CompressedImgPublisher(self.cam, '/trr_vision/lane/image_debug')
        # Model publishing
        self.lane_model_pub = trr_rpu.LaneModelPublisher('/trr_vision/lane/detected_model', who='trr_vision_lane_node')
        self.lane_model = trru.LaneModel()
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_vision_laneConfig, self.cfg_callback)
        self.status_pub = trr_rpu.VisionLaneStatusPublisher('/trr/vision/lane/status', who='trr_vision_lane_node')
        # start image subscription only here
        self.start()

        
    def cfg_callback(self, config, level):
        rospy.loginfo("  Reconfigure Request:")
        try:
            self.pipeline.thresholder.set_threshold(config['mask_threshold'])
        except AttributeError: pass
        self.pipeline.display_mode = config['display_mode']
        return config

    def pipe_cbk(self):
        if self.pipeline.lane_model.is_valid():
            self.lane_model_pub.publish(self.pipeline.lane_model)
            
    def periodic(self):
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
            #self.publish_3Dmarkers()
        self.status_pub.publish(self.pipeline)
    
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
