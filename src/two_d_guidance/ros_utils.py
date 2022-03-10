import math, numpy as np, scipy, rospy, geometry_msgs.msg, nav_msgs.msg, tf, pickle

#import smocap.utils

def list_of_xyz(p): return [p.x, p.y, p.z]
def array_of_xyz(p): return np.array(list_of_xyz(p))
def list_of_xyzw(q): return [q.x, q.y, q.z, q.w]

class RobotNotLocalizedException(Exception):
    pass
class RobotLostException(Exception):
    pass

class SmocapListener:
    def __init__(self, topic='/smocap/est_marker'):
        rospy.Subscriber(topic, geometry_msgs.msg.PoseWithCovarianceStamped, self.smocap_cbk)
        self.ts = None
        self.pose = None
        self.vel = None

    def smocap_cbk(self, msg):
        if self.pose is not None:
            p1 = array_of_xyz(self.pose.position)
            p2 = array_of_xyz(msg.pose.pose.position)
            dt = msg.header.stamp.to_sec() - self.ts
            self.vel = np.linalg.norm((p2-p1)/dt)
        self.pose = msg.pose.pose
        self.ts = msg.header.stamp.to_sec()

    def get_loc_and_yaw(self, max_delay=0.2):
        if self.ts is None:
            raise RobotNotLocalizedException
        if rospy.Time.now().to_sec() - self.ts > max_delay:
            raise RobotLostException
        l = array_of_xyz(self.pose.position)[:2]
        y = tf.transformations.euler_from_quaternion(list_of_xyzw(self.pose.orientation))[2]
        return l, y

class RobotPoseListener(SmocapListener): pass

class GazeboTruthListener:
    def __init__(self, topic='/homere/base_link_truth'):
        rospy.Subscriber(topic, nav_msgs.msg.Odometry, self.truth_cbk)
        self.ts = None
        self.pose = None
        self.vel = None

    def truth_cbk(self, msg):
        #print msg.pose.pose.position
        #print msg.twist.twist.linear
        self.pose = msg.pose.pose
        self.twist = msg.twist.twist
        self.vel = abs(self.twist.linear.x)
        self.ts = msg.header.stamp.to_sec()

    def get_loc_and_yaw(self, max_delay=0.2):
        if self.ts is None:
            raise RobotNotLocalizedException
        if rospy.Time.now().to_sec() - self.ts > max_delay:
            raise RobotLostException
        l = array_of_xyz(self.pose.position)[:2]
        y = tf.transformations.euler_from_quaternion(list_of_xyzw(self.pose.orientation))[2]
        return l, y
 
    def get_bl_to_world_T(self, max_delay=0.2):
        if self.ts is None:
            raise RobotNotLocalizedException
        if rospy.Time.now().to_sec() - self.ts > max_delay:
            raise RobotLostException
        t, q = array_of_xyz(self.pose.position), list_of_xyzw(self.pose.orientation)
        T = smocap.utils.T_of_t_q(t, q)
        return T

class RobotOdomListener(GazeboTruthListener): pass
