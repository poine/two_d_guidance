#!/usr/bin/env python
import sys
import rospy, dynamic_reconfigure.server, cv2
import pdb

import two_d_guidance.trr.vision.utils as trr_vu
import two_d_guidance.trr.rospy_utils as trr_rpu
import two_d_guidance.trr.vision.traffic_light as trr_tl

import two_d_guidance.cfg.trr_vision_traffic_lightConfig

class Node(trr_rpu.TrrSimpleVisionPipeNode):
    def __init__(self):
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, trr_tl.TrafficLightPipeline, self.pipe_cbk)
        self.traffic_light_pub = trr_rpu.TrrTrafficLightPublisher()
        #self.img_pub = trr_rpu.ImgPublisher(self.cam, '/trr_vision/traffic_light/image_debug')
        self.img_pub = trr_rpu.CompressedImgPublisher(self.cam, '/trr_vision/traffic_light/image_debug')
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_vision_traffic_lightConfig, self.cfg_callback)

    def cfg_callback(self, cfg, level):
        rospy.loginfo("  Reconfigure Request:")
        #print cfg, level
        self.pipeline.set_debug_display(cfg['display_mode'], cfg['show_hud'])
        #pdb.set_trace()
        self.pipeline.set_green_mask_params(cfg.g_hc, cfg.g_hs, cfg.g_smin, cfg.g_smax, cfg.g_vmin, cfg.g_vmax)

        self.pipeline.set_red_mask_params(cfg.r_hc, cfg.r_hs, cfg.r_smin, cfg.r_smax, cfg.r_vmin, cfg.r_vmax)

        x, y, dx, dy = cfg.roi_xc, cfg.roi_yc, cfg.roi_dx, cfg.roi_dy
        tl, br = (x-dx/2, y-dy/2), (x+dx/2, y+dy/2)
        self.pipeline.set_roi(tl, br)

        return cfg
    
    def periodic(self):
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)

    def pipe_cbk(self):
        self.traffic_light_pub.publish(self.pipeline)

def main(args):
    name = 'trr_vision_traffic_light_node'
    rospy.init_node(name)
    rospy.loginfo('{} starting'.format(name))
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run()


if __name__ == '__main__':
    main(sys.argv)
