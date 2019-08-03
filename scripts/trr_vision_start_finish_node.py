#!/usr/bin/env python

import sys, numpy as np, rospy,  dynamic_reconfigure.server, sensor_msgs.msg
import cv2#, cv_bridge
import pdb

import smocap.rospy_utils
import two_d_guidance.trr_utils as trru
import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_rospy_utils as trr_rpu
import two_d_guidance.trr.vision.start_finish as trr_vsf

class MarkerPublisher:
    def __init__(self):
        pass
    
'''

'''
class Node(trr_rpu.TrrSimpleVisionPipeNode):

    def __init__(self):
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, trr_vsf.StartFinishDetectPipeline, self.pipe_cbk)

        self.start_finish_pub = trr_rpu.TrrStartFinishPublisher()
        self.img_pub = trr_rpu.ImgPublisher(self.cam, '/trr_vision/start_finish/image_debug')
        self.start_ctr_pub = trr_rpu.ContourPublisher(frame_id=self.ref_frame,
                                                      topic='/trr_vision/start_finish/start_contour', rgba=(0.,1.,0.,1.))
        self.finish_ctr_pub = trr_rpu.ContourPublisher(frame_id=self.ref_frame,
                                                      topic='/trr_vision/start_finish/finish_contour', rgba=(1.,0.,0.,1.))
        
    def pipe_cbk(self):
        self.start_finish_pub.publish(self.pipeline)

    def periodic(self):
        self.img_pub.publish(self.pipeline, self.cam)
        self.publish_3Dmarkers()

    def publish_3Dmarkers(self):
        if self.pipeline.green_ctr_lfp is not None:
            self.start_ctr_pub.publish(self.pipeline.green_ctr_lfp)
        if self.pipeline.red_ctr_lfp is not None:
            self.finish_ctr_pub.publish(self.pipeline.red_ctr_lfp)


def main(args):
    name = 'trr_vision_start_stop_detect_node'
    rospy.init_node(name)
    rospy.loginfo('{name} starting')
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run()


if __name__ == '__main__':
    main(sys.argv)
