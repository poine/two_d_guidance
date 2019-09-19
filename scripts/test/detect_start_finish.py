#!/usr/bin/env python
import os, time
import numpy as np
import cv2
import rosbag, cv_bridge
import pdb

import two_d_guidance.trr.vision.utils as trr_vu

        
#PH inclure start_finish et utiliser StartFinishDetectPipeline
        
def work(sfd, img):
    sfd.process_image(img)
    
def update_display(sfd, img):
    cv2.imshow('orig', img)
    img2 = sfd.draw(img)
    if img2 is not None:
        cv2.imshow('Masks', img2)

def test_on_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    work(img)
    update_display(img)
    cv2.waitKey(0)

def test_on_bag(bag_path, img_topic, odom_topic, sleep=False, display=True):
    bag, bridge = rosbag.Bag(bag_path, "r"), cv_bridge.CvBridge()
    sfd = trr_vu.StartFinishDetector()
    durations, last_img_t = [], None
    times = []
    odom_times, odom_vlins, odom_vangs = [], [], []
    img_idx = -1
    for topic, msg, img_t in bag.read_messages(topics=[img_topic, odom_topic]):
        if topic == img_topic:
            img_idx += 1
            img_dt = 0.01 if last_img_t is None else (img_t-last_img_t).to_sec()
            cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            work(sfd, cv_img)
            if display:
                update_display(sfd, cv_img)
                cv2.waitKey(1)
                #cv2.waitKey(0)
            last_img_t = img_t
            time_to_sleep = img_dt if sleep else 0
            time.sleep(time_to_sleep)
        elif topic == odom_topic:
            odom_times.append(msg.header.stamp.to_sec())
            odom_vlins.append(msg.twist.twist.linear.x)
            odom_vangs.append(msg.twist.twist.angular.z)
        




    
if __name__ == '__main__':
    mode_img, mode_bag = range(2)
    
    prefix_dir = '/home/poine'
    mode = mode_bag
    if mode == mode_img:
        img_filename = 'work/robot_data/caroline/gazebo/start_line_05.png'

        img_path = os.path.join(prefix_dir, img_filename)

        test_on_img(img_path)

    elif mode == mode_bag:

        #bag_filename = '2019-09-10-14-00-00.bag' # Christine Vedrines 3 tours, nuageux
        # Camera road + odom + imu
        bag_filename = '2019-09-16-18-40-00.bag' # Christine Vedrines 1 tour, ombres longues, vitesse constante 2.5
        # bag_filename = '2019-09-16-18-43-00.bag' # Christine Vedrines, ombres longues, acceleration en lignes droites a 4 puis 5

        img_topic, odom_topic = '/camera_road_front/image_raw', '/oscar_ackermann_controller/odom'
        bag_path = os.path.join(prefix_dir, bag_filename)
        test_on_bag(bag_path, img_topic, odom_topic, sleep=True)
