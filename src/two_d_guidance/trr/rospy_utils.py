import sys, numpy as np
import cv2, cv_bridge, tf
import rospy, sensor_msgs.msg, visualization_msgs.msg, geometry_msgs.msg, nav_msgs.msg
import pdb

#
# A bunch of utilities to deal with ROS
#   - publishers and listeners for our messages
#   - skeletons for common nodes
#


import smocap # for cameras, needs to go
import smocap.rospy_utils
import two_d_guidance.msg
import trr.msg

def list_of_xyz(p): return [p.x, p.y, p.z]
def array_of_xyz(p): return np.array(list_of_xyz(p))
def list_of_xyzw(q): return [q.x, q.y, q.z, q.w]
def msgPoint(x, y, z): p = geometry_msgs.msg.Point(); p.x=x; p.y=y;p.z=z; return p

class NoRXMsgException(Exception): pass
class RXMsgTimeoutException(Exception): pass

class SimplePublisher(rospy.Publisher):
    def __init__(self, topic, msg_class, what, qs=1):
        rospy.loginfo(' {} publishing on {}'.format(what, topic))
        rospy.Publisher.__init__(self, topic, msg_class, queue_size=qs)
        self.msg_class = msg_class
        
class SimpleSubscriber:
    def __init__(self, topic, msg_class, what, timeout=0.5, user_cbk=None):
        self.sub = rospy.Subscriber(topic, msg_class, self.msg_callback, queue_size=1)
        rospy.loginfo(' {} subscribed to {}'.format(what, topic))
        self.timeout, self.user_cbk = timeout, user_cbk
        self.msg = None
        
    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()
        if self.user_cbk is not None: self.user_cbk(msg)

    def get(self):
        if self.msg is None:
            raise NoRXMsgException
        if (rospy.get_rostime()-self.last_msg_time).to_sec() > self.timeout:
            raise RXMsgTimeoutException
        return self.msg

#
# Race Manager Status
#
def RaceManagerStatusStr(_i): return ['Staging', 'Ready', 'Racing', 'Finished', 'JoinStart'][_i]
class RaceManagerStatusPublisher(SimplePublisher):
    def __init__(self, topic='trr_race_manager/status'):
        SimplePublisher.__init__(self, topic, trr.msg.RaceManagerStatus, 'race manager') 

    def publish(self, src):
        msg = trr.msg.RaceManagerStatus()
        msg.mode = src.mode
        msg.cur_lap = src.cur_lap
        msg.tot_lap = src.nb_lap
        msg.lap_times = src.lap_times
        SimplePublisher.publish(self, msg)


class RaceManagerStatusSubscriber(SimpleSubscriber):
    def __init__(self, topic='trr_race_manager/status', what='N/A', timeout=0.1, user_callback=None):
        SimpleSubscriber.__init__(self, topic, trr.msg.RaceManagerStatus, what, timeout, user_callback)

    def get(self):
        msg = SimpleSubscriber.get(self) # raise exceptions
        return msg.mode, msg.cur_lap, msg.tot_lap, msg.lap_times


#
# Guidance Status
#
class GuidanceStatusPublisher(SimplePublisher):
    def __init__(self, topic='trr/guidance/status', what='N/A', timeout=0.1, user_callback=None):
        SimplePublisher.__init__(self, topic, two_d_guidance.msg.FLGuidanceStatus, what) # FIXME trr.msg.GuidanceStatus

    def publish(self, model):
        msg = two_d_guidance.msg.FLGuidanceStatus()
        msg.guidance_mode = model.mode
        msg.poly = model.lane.coefs
        msg.lookahead_dist = model.lookahead_dist
        msg.lookahead_time = model.lookahead_time
        msg.carrot_x, msg.carrot_y = model.carrot
        msg.R = model.R
        msg.lin_sp, msg.ang_sp = model.lin_sp, model.ang_sp
        SimplePublisher.publish(self, msg)

class GuidanceStatusSubscriber(SimpleSubscriber):
    def __init__(self, topic='trr_guidance/status', what='N/A', timeout=0.1, user_callback=None):
        SimpleSubscriber.__init__(self, topic, two_d_guidance.msg.FLGuidanceStatus, what, timeout, user_callback)
        
    # def get(self):
    #     msg = SimpleSubscriber.get(self) # raise exceptions
    #     return msg

    
#
# StartFinish
#
class TrrStartFinishPublisher(SimplePublisher):
    def __init__(self, topic='trr_vision/start_finish/status'):
        SimplePublisher.__init__(self, topic, two_d_guidance.msg.TrrStartFinish, 'start finish')
        #rospy.loginfo(' publishing start finish status on ({})'.format(topic))
        #self.pub = rospy.Publisher(topic, two_d_guidance.msg.TrrStartFinish, queue_size=1)

    def publish(self, pl):
        msg = two_d_guidance.msg.TrrStartFinish()
        if pl.ss_dtc.sees_start():
            for (x, y, z) in pl.start_ctr_lfp:
               msg.start_points.append(msgPoint(x, y, z))
        if pl.ss_dtc.sees_finish():
            for (x, y, z) in pl.finish_ctr_lfp:
               msg.finish_points.append(msgPoint(x, y, z))
        msg.dist_to_finish = pl.dist_to_finish
        msg.dist_to_start = pl.dist_to_start
        SimplePublisher.publish(self, msg)

        
class TrrStartFinishSubscriber(SimpleSubscriber):
    def __init__(self, topic='trr_vision/start_finish/status', what='N/A', timeout=0.1, user_callback=None):
        SimpleSubscriber.__init__(self, topic, two_d_guidance.msg.TrrStartFinish, what, timeout, user_callback)

    def get(self):
        msg = SimpleSubscriber.get(self)
        def array_of_pts(pts): return [[p.x, p.y, p.z] for p in  pts]
        return array_of_pts(self.msg.start_points), array_of_pts(self.msg.finish_points), self.msg.dist_to_start, self.msg.dist_to_finish 
    def viewing_finish(self):
        try:
            msg = SimpleSubscriber.get(self)
        except NoRXMsgException, RXMsgTimeoutException:
            return False
        return self.msg.finish_points
#
# StateEstimation
#               
class TrrStateEstimationPublisher(SimplePublisher):
    def __init__(self, topic='trr_state_est/status'):
        SimplePublisher.__init__(self, topic, two_d_guidance.msg.TrrStateEst, 'state estimation')

    def publish(self, model):
        msg = self.msg_class()
        msg.s = model.sn
        msg.idx_s = model.idx_sn
        msg.v = model.v
        msg.dist_to_finish = model.predicted_dist_to_finish
        msg.dist_to_start  = model.predicted_dist_to_start
        SimplePublisher.publish(self, msg)

class TrrStateEstimationSubscriber(SimpleSubscriber):
    def __init__(self, topic='trr_state_est/status', what='unkown', timeout=0.1, user_callback=None):
        SimpleSubscriber.__init__(self, topic, two_d_guidance.msg.TrrStateEst, what, timeout, user_callback)

    def get(self):
        msg = SimpleSubscriber.get(self)
        return msg.s, msg.idx_s, msg.v, msg.dist_to_start, msg.dist_to_finish 

#
# Lanes
# 
class LaneModelPublisher(SimplePublisher):
    def __init__(self, topic, who='N/A'):
        SimplePublisher.__init__(self, topic, two_d_guidance.msg.LaneModel, who)

    def publish(self, lm):
        msg = two_d_guidance.msg.LaneModel()
        msg.poly = lm.coefs
        SimplePublisher.publish(self, msg)

        
class LaneModelSubscriber(SimpleSubscriber):
    def __init__(self, topic, what='', timeout=0.1, user_cbk=None):
        SimpleSubscriber.__init__(self, topic, two_d_guidance.msg.LaneModel, what, timeout, user_cbk)

    def get(self, lm):
        msg = SimpleSubscriber.get(self) # raise exceptions
        lm.coefs = self.msg.poly
        lm.set_valid(True)


#
# Vision Status
#
class VisionLaneStatusPublisher(SimplePublisher):
    def __init__(self, topic, who='N/A'):
        SimplePublisher.__init__(self, topic, trr.msg.VisionLaneStatus, who)

    def publish(self, pipe):
        msg = trr.msg.VisionLaneStatus()
        msg.fps = pipe.lp_fps
        msg.cpu_t = pipe.lp_proc
        msg.idle_t = pipe.idle_t
        msg.skipped = pipe.skipped_frames
        SimplePublisher.publish(self, msg)


        
#
# Odometry
#               
class OdomListener(SimpleSubscriber):
    def __init__(self, topic='/caroline/diff_drive_controller/odom', what='N/A', timeout=0.1, user_cbk=None):
        SimpleSubscriber.__init__(self, topic, nav_msgs.msg.Odometry, what, timeout, user_cbk)
        
    # def get(self):
    #     msg = SimpleSubscriber.get(self)
    #     return msg.twist.twist.linear.x, msg.twist.twist.angular.z

    def get_vel(self):
        msg = SimpleSubscriber.get(self)
        return msg.twist.twist.linear.x, msg.twist.twist.angular.z

    def get_pose(self):
        msg = SimpleSubscriber.get(self)
        return np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def get_2Dpose(self):
        msg = SimpleSubscriber.get(self)
        return np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def get_loc_and_yaw(self):
        msg = SimpleSubscriber.get(self)
        l = array_of_xyz(msg.pose.pose.position)[:2]
        y = tf.transformations.euler_from_quaternion(list_of_xyzw(msg.pose.pose.orientation))[2]
        return l, y
    
#
# Traffic light
#   
class TrrTrafficLightPublisher:
    def __init__(self, topic='trr_vision/traffic_light/status'):
        rospy.loginfo(' publishing traffic light status on ({})'.format(topic))
        self.pub = rospy.Publisher(topic, two_d_guidance.msg.TrrTrafficLight, queue_size=1)
    def publish(self, pl):
        msg = two_d_guidance.msg.TrrTrafficLight()
        msg.red, msg.yellow, msg.green = pl.get_light_status()
        self.pub.publish(msg)

class TrrTrafficLightSubscriber:
    def __init__(self, topic='trr_vision/traffic_light/status'):
        self.sub = rospy.Subscriber(topic, two_d_guidance.msg.TrrTrafficLight, self.msg_callback, queue_size=1)
        rospy.loginfo(' subscribed to ({})'.format(topic))
        self.msg = None
        self.timeout = 0.5     

    def msg_callback(self, msg):
        self.msg = msg
        self.last_msg_time = rospy.get_rostime()

    def get(self):
        if self.msg is not None and (rospy.get_rostime()-self.last_msg_time).to_sec() < self.timeout:
            return self.msg.red, self.msg.yellow, self.msg.green
        else: return False, False, False

#
# Images
#
       
class ImgPublisher:
    def __init__(self, cam, img_topic = "/trr_vision/start_finish/image_debug"):
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.Image, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        
    def publish(self, producer, cam, encoding="rgb8"):
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(producer.draw_debug(cam), encoding))

class CompressedImgPublisher:
    def __init__(self, cam, img_topic):
        img_topic = img_topic + "/compressed"
        rospy.loginfo(' publishing image on ({})'.format(img_topic))
        self.image_pub = rospy.Publisher(img_topic, sensor_msgs.msg.CompressedImage, queue_size=1)
        
    def publish(self, model, data):
        img_rgb = model.draw_debug(data)
        self.publish2(img_rgb)
        
    def publish2(self, img_rgb):
        img_bgr =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        msg = sensor_msgs.msg.CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_bgr)[1]).tostring()
        self.image_pub.publish(msg)
        
        
#
# Markers
#       


class LaneModelMarkerPublisher:
    def __init__(self, topic, ref_frame="nono_0/base_link_footprint",
                 color=(1., 0., 1., 0.)):
        self.pub_lane = rospy.Publisher(topic, visualization_msgs.msg.Marker, queue_size=1)
        rospy.loginfo(' publishing lane model markers on ({})'.format(topic))
        self.lane_msg = visualization_msgs.msg.Marker()
        self.lane_msg.header.frame_id = ref_frame
        self.lane_msg.type = self.lane_msg.LINE_STRIP
        self.lane_msg.action = self.lane_msg.ADD
        self.lane_msg.id = 0
        self.lane_msg.text = 'lane'
        s = self.lane_msg.scale; s.x, s.y, s.z = 0.01, 0.2, 0.2
        c = self.lane_msg.color; c.a, c.r, c.g, c.b = color
        o = self.lane_msg.pose.orientation; o.x, o.y, o.z, o.w = 0, 0, 0, 1

    def publish(self, lm, l0=0.6, l1=1.8):
        #pts = [[0, 0, 0], [0.1, 0, 0], [0.2, 0.1, 0]]
        try:
            l0, l1 = lm.x_min, lm.x_max
        except AttributeError: pass
        pts = [[x, lm.get_y(x), 0] for x in np.linspace(l0, l1, 10)]
        self.lane_msg.points = []
        for p in pts:
            _p = geometry_msgs.msg.Point()
            _p.x, _p.y, _p.z = p
            self.lane_msg.points.append(_p)
        self.pub_lane.publish( self.lane_msg)  


class ContourPublisher:
    def __init__(self, frame_id='caroline/base_link_footprint', topic='/contour',
                 contour_blf=None, rgba=(1.,1.,0.,1.)):
        self.frame_id, self.rgba = frame_id, rgba
        self.pub = rospy.Publisher(topic , visualization_msgs.msg.MarkerArray, queue_size=1)
        rospy.loginfo(' publishing contour on ({})'.format(topic))
        self.msg = visualization_msgs.msg.MarkerArray()
        # if we were given a contour, setup our marker message
        if contour_blf is not None:
            self.msg.markers=[self.get_marker(contour_blf, self.rgba)]

    def get_marker(self, contour_blf, rgba=(1.,1.,0.,1.), closed=True):
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = self.frame_id
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        marker.id = 0
        marker.text = 'foo'
        s = marker.scale; s.x, s.y, s.z = 0.01, 0.2, 0.2
        c = marker.color; c.r, c.g, c.b, c.a = rgba
        o = marker.pose.orientation; o.w, o.x, o.y, o.z = 1, 0, 0, 0
        for x, y, z in contour_blf:
            marker.points.append(msgPoint(x, y, z))
        if closed:
            x, y, z = contour_blf[0]
            marker.points.append(msgPoint(x, y, z))
        return marker

    def publish(self, contour_blf=None):
        if contour_blf is not None:
            self.msg.markers=[self.get_marker(contour_blf, self.rgba)]
        self.pub.publish(self.msg)
        






class TrrSimpleVisionPipeNode:
    def __init__(self, pipeline_class, pipe_cbk=None):
        robot_name = rospy.get_param('~robot_name', '')
        def prefix(robot_name, what): return what if robot_name == '' else '{}/{}'.format(robot_name, what)
        self.cam_names = rospy.get_param('~cameras', prefix(robot_name, 'camera_road_front')).split(',')
        self.ref_frame = rospy.get_param('~ref_frame', prefix(robot_name, 'base_link_footprint'))

        self.cam_sys = smocap.rospy_utils.CamSysRetriever().fetch(self.cam_names, fetch_extrinsics=True, world=self.ref_frame)
        self.cam = self.cam_sys.cameras[0]; self.cam.set_undistortion_param(alpha=1.)

        self.pipeline = pipeline_class(self.cam, robot_name)

    def start(self):
        self.cam_lst = smocap.rospy_utils.CamerasListener(cams=self.cam_names, cbk=self.on_image)

    # we get a bgr8 image as input
    def on_image(self, img_bgr, (cam_idx, stamp, seq)):
        self.pipeline.process_image(img_bgr, self.cam_sys.cameras[cam_idx], stamp, seq)
        if self.pipe_cbk is not None: self.pipe_cbk()
        
    def run(self, low_freq=10):
        rate = rospy.Rate(low_freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass



class DebugImgPublisher:
    def __init__(self, cam_name, topic_sink):
        self.image_pub = CompressedImgPublisher(cam=None, img_topic=topic_sink)

        self.img, self.compressed_img = None, None
        img_src_topic = cam_name + '/image_raw/compressed'
        self.img_sub = rospy.Subscriber(img_src_topic, sensor_msgs.msg.CompressedImage, self.img_cbk,  queue_size = 1)
        rospy.loginfo(' subscribed to ({})'.format(img_src_topic))

    def img_cbk(self, msg):
        self.compressed_img = np.fromstring(msg.data, np.uint8)

    def publish(self, model, user_data):
        n_subscriber = self.image_pub.image_pub.get_num_connections()
        # don't bother drawing and publishing when no one is listening
        if n_subscriber <= 0: return
        if self.compressed_img is not None:
            self.img_bgr = cv2.imdecode(self.compressed_img, cv2.IMREAD_COLOR)
            self._draw(self.img_bgr, model, user_data)
            self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            #img_rgb = self.img[...,::-1] # rgb = bgr[...,::-1] OpenCV image to Matplotlib
            self.image_pub.publish2(self.img_rgb)



class PeriodicNode:

    def __init__(self, name):
        rospy.init_node(name)
    
    def run(self, freq):
        rate = rospy.Rate(freq)
        try:
            while not rospy.is_shutdown():
                self.periodic()
                rate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass
