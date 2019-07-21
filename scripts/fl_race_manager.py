#!/usr/bin/env python
import os, sys, roslib, rospy, rospkg, rostopic
import dynamic_reconfigure.client
#from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
#from geometry_msgs.msg import PolygonStamped, Point32

class Node:

    def __init__(self):
        self.low_freq = 20
        #self.pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=1)
        client_name = "follow_line_guidance"
        client = dynamic_reconfigure.client.Client(client_name, timeout=30, config_callback=self.callback)
        rospy.loginfo(' client_name: {}'.format(client_name))
        #client.update_configuration({"vel_sp":0.9})
        client.update_configuration({"guidance_mode":2})
                             
    def callback(self, config):
        rospy.loginfo("Config set {}".format(config))
   
    def periodic(self):
        # obstacle_msg = ObstacleArrayMsg() 
        # obstacle_msg.header.stamp = rospy.Time.now()
        # obstacle_msg.header.frame_id = "odom" # CHANGE HERE: odom/map
        # #Add line obstacle
        # obstacle_msg.obstacles.append(ObstacleMsg())
        # obstacle_msg.obstacles[0].id = 0
        # line_start = Point32()
        # line_start.x = -1.5
        # line_start.y = -1.
        # line_end = Point32()
        # line_end.x = -1.5
        # line_end.y = 1
        # obstacle_msg.obstacles[0].polygon.points = [line_start, line_end]
        # self.pub.publish(obstacle_msg)
        pass
    
    def run(self):
        rate = rospy.Rate(self.low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass

        
def main(args):
  rospy.init_node('race_manager_node')
  Node().run()


if __name__ == '__main__':
    main(sys.argv)
