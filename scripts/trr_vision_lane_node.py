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
import two_d_guidance.trr.vision.lane_2 as trr_l2 # Contour detection in thresholded gray camera image with bird eye
import two_d_guidance.trr.vision.lane_3 as trr_l3 # Contour detection in thresholded gray camera image with bird eye
import two_d_guidance.trr.vision.lane_4 as trr_l4 # In progress

import two_d_guidance.cfg.trr_vision_laneConfig

class BirdEyePublisher(trr_rpu.ContourPublisher):
    def __init__(self, frame_id, be_param, topic='trr_vision/lane/bird_eye'):
        trr_rpu.ContourPublisher.__init__(self, frame_id, topic, be_param.corners_be_blf, rgba=(1.,0.,0.,1.))
        
'''

'''
class Node(trr_rpu.TrrSimpleVisionPipeNode):

    def __init__(self):
        pipe_type = 2
        pipe_classes = [trr_l1.Contour1Pipeline, trr_l2.Contour2Pipeline, trr_l3.Contour3Pipeline ]
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, pipe_classes[pipe_type], self.pipe_cbk)
        try:
            robot_name = rospy.get_param('~robot_name', 'christine')#'caroline'
            param = trr_vu.NamedBirdEyeParam(robot_name)
            rospy.loginfo("  using {} bird eye".format(robot_name))
            self.pipeline.bird_eye.set_param(param)
            self.pipeline.bird_eye.compute_H(self.cam)
            self.pipeline.bird_eye.compute_H2(self.cam)
        except AttributeError: rospy.loginfo("  pipeline has no bird eye")
        
        roi_y_min = rospy.get_param('~roi_y_min', 0)
        tl, br = (0, roi_y_min), (self.cam.w, self.cam.h)
        self.pipeline.set_roi(tl, br) 
        # Image publishing
        self.img_pub = trr_rpu.CompressedImgPublisher(self.cam, '/trr_vision/lane/image_debug')
        # Markers publishing
        #self.cont_pub = trr_rpu.ContourPublisher(topic='/trr_vision/lane/detected_contour_markers', frame_id=self.ref_frame)
        #self.fov_pub = smocap.rospy_utils.FOVPublisher(self.cam_sys, self.ref_frame, '/trr_vision/lane/fov')
        #try:
        #    self.be_pub = BirdEyePublisher(self.ref_frame, self.pipeline.bird_eye.param, '/trr_vision/lane/bird_eye')
        #except AttributeError:
        #    self.be_pub = None
        #self.lane_model_marker_pub = trr_rpu.LaneModelMarkerPublisher('/trr_vision/lane/detected_model_markers', ref_frame=self.ref_frame)
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
            
    #def publish_3Dmarkers(self):
        #self.fov_pub.publish()
        #if self.be_pub is not None:
            #self.be_pub.publish()
            #if self.pipeline.bird_eye.cnt_fp is not None:
            #    self.cont_pub.publish(self.pipeline.bird_eye.cnt_fp)
        #self.lane_model_marker_pub.publish(self.pipeline.lane_model)

            
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
