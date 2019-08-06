#!/usr/bin/env python
import sys
import rospy, dynamic_reconfigure.server, cv2
import pdb

import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_rospy_utils as trr_rpu
import two_d_guidance.trr.vision.traffic_light as trr_tl

import two_d_guidance.cfg.trr_vision_traffic_lightConfig

class Node(trr_rpu.TrrSimpleVisionPipeNode):
    def __init__(self):
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, trr_tl.TrafficLightPipeline, self.pipe_cbk)
        self.traffic_light_pub = trr_rpu.TrrTrafficLightPublisher()
        self.img_pub = trr_rpu.ImgPublisher(self.cam, '/trr_vision/traffic_light/image_debug')
        self.cfg_srv = dynamic_reconfigure.server.Server(two_d_guidance.cfg.trr_vision_traffic_lightConfig, self.cfg_callback)

    def cfg_callback(self, config, level):
        rospy.loginfo("  Reconfigure Request:")
        #print config, level
        self.pipeline.display_mode = config['display_mode']
        #pdb.set_trace()
        g_sens, g_smin, g_smax, g_vmin, g_vmax = config.gh_sens, config.gs_min, config.gs_max, config.gv_min, config.gv_max
        self.pipeline.set_green_mask_params(g_sens, g_smin, g_smax, g_vmin, g_vmax)
        r_hsens, r_smin, r_smax, r_vmin, r_vmax = config.rh_sens, config.rs_min, config.rs_max, config.rv_min, config.rv_max
        self.pipeline.set_red_mask_params(r_hsens, r_smin, r_smax, r_vmin, r_vmax)
        #tl_x, tl_y, br_x, br_y = config.tl_x, config.tl_y, config.br_x, config.br_y
        self.pipeline.set_roi((config.roi_tl_x, config.roi_tl_y), (config.roi_br_x, config.roi_br_y))
        return config
    
    def periodic(self):
        if self.pipeline.display_mode != self.pipeline.show_none:
            self.img_pub.publish(self.pipeline, self.cam)

    def pipe_cbk(self):
        self.traffic_light_pub.publish(self.pipeline)
        pass

def main(args):
    name = 'trr_vision_traffic_light_node'
    rospy.init_node(name)
    rospy.loginfo('{name} starting')
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run()


if __name__ == '__main__':
    main(sys.argv)
