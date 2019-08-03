#!/usr/bin/env python
import sys
import rospy, cv2

import two_d_guidance.trr_vision_utils as trr_vu
import two_d_guidance.trr_rospy_utils as trr_rpu
import two_d_guidance.trr.vision.traffic_light as trr_tl


class Node(trr_rpu.TrrSimpleVisionPipeNode):
    def __init__(self):
        trr_rpu.TrrSimpleVisionPipeNode.__init__(self, trr_tl.TrafficLightPipeline, self.pipe_cbk)
        #self.traffic_light_pub = trr_rpu.TrrTrafficLightPublisher()
        self.img_pub = trr_rpu.ImgPublisher(self.cam, '/trr_vision/traffic_light/image_debug')
    
    def periodic(self):
        self.img_pub.publish(self.pipeline, self.cam)

    def pipe_cbk(self):
        #self.start_finish_pub.publish(self.pipeline)
        pass

def main(args):
    name = 'trr_vision_traffic_light_node'
    rospy.init_node(name)
    rospy.loginfo('{name} starting')
    rospy.loginfo('  using opencv version {}'.format(cv2.__version__))
    Node().run()


if __name__ == '__main__':
    main(sys.argv)
