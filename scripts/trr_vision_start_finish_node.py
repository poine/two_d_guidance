#!/usr/bin/env python

import sys, numpy as np, rospy,  dynamic_reconfigure.server, sensor_msgs.msg
import cv2#, cv_bridge
import pdb

import smocap.rospy_utils
import two_d_guidance.trr.utils as trru
import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.vision.start_finish as trr_vsf

import two_d_guidance.cfg.trr_vision_start_finishConfig

class MarkerPublisher:
    def __init__(self, ref_frame):
        self.start_ctr_pub = trr_rpu.ContourPublisher(frame_id=ref_frame,
                                                      topic='/trr_vision/start_finish/start_contour', rgba=(0.,1.,0.,1.))
        self.finish_ctr_pub = trr_rpu.ContourPublisher(frame_id=ref_frame,
                                                       topic='/trr_vision/start_finish/finish_contour', rgba=(1.,0.,0.,1.))
    def publish(self, pipeline):
        if pipeline.start_ctr_lfp is not None:
            self.start_ctr_pub.publish(pipeline.start_ctr_lfp)
        if pipeline.finish_ctr_lfp is not None:
            self.finish_ctr_pub.publish(pipeline.finish_ctr_lfp)
    
'''

'''
class Node(trr_rpu.TrrSimpleVisionPipeNode):

    def __init__(self):
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, trr_vsf.StartFinishDetectPipeline, self.pipe_cbk)

        self.start_finish_pub = trr_rpu.TrrStartFinishPublisher()
        #self.img_pub = trr_rpu.ImgPublisher(self.cam, '/trr_vision/start_finish/image_debug')
        self.img_pub = trr_rpu.CompressedImgPublisher(self.cam, '/trr_vision/start_finish/image_debug')
        self.marker_pub = MarkerPublisher(self.ref_frame)
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_vision_start_finishConfig, self.cfg_callback)

    def cfg_callback(self, cfg, level):
        rospy.loginfo("  Reconfigure Request:")
        #print cfg, level
        self.pipeline.set_debug_display(cfg['display_mode'], cfg['show_hud'])
        self.pipeline.set_green_mask_params(cfg.g_hc, cfg.g_hs, cfg.g_smin, cfg.g_smax, cfg.g_vmin, cfg.g_vmax, cfg.g_gthr)
        self.pipeline.set_red_mask_params(cfg.r_hc, cfg.r_hs, cfg.r_smin, cfg.r_smax, cfg.r_vmin, cfg.r_vmax, cfg.r_gthr)
        yt, xbr, ybr = cfg.roi_yt, self.cam.w, self.cam.h
        tl, br = (0, yt), (xbr, ybr)
        self.pipeline.set_roi(tl, br)
        
        return cfg
    
    def pipe_cbk(self):
        self.start_finish_pub.publish(self.pipeline)

    def periodic(self):
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)
            self.marker_pub.publish(self.pipeline)



def main(args):
    name = 'trr_vision_start_finish_node'
    rospy.init_node(name)
    rospy.loginfo('{name} starting')
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run()

if __name__ == '__main__':
    main(sys.argv)
