#!/usr/bin/env python
import time, numpy as np, matplotlib.pyplot as plt, cv2
import rosbag, cv_bridge
import pdb

import smocap
import fl_vision_utils as flvu


def test_on_img(pipe, cam, img):
    pipe.process_image(img, cam)
    out_img = pipe.draw_debug(cam)
    cv2.imshow('in', img); cv2.imshow('out', out_img)
    cv2.waitKey(0)


def test_on_bag(pipe, cam, bag_path, img_topic='/camera1/image_raw' ):
    bag, bridge = rosbag.Bag(bag_path, "r"), cv_bridge.CvBridge()
    durations = []
    for topic, msg, t in bag.read_messages(topics=[img_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        #print cv_img.dtype
        #if cv_img.dtype == np.uint8:
        #    cv_img = cv_img.astype(np.float32)/255.
        pipe.process_image(cv_img, cam)
        durations.append(pipe.last_duration)
        print('{:.3f}s ({:.1f}hz)'.format(pipe.last_duration, 1./pipe.last_duration ))
        out_img = pipe.draw_debug(cam)
        cv2.imshow('out', out_img)
        cv2.waitKey(10)
    freqs = 1./np.array(durations); _mean, _std, _min, _max = np.mean(freqs), np.std(freqs), np.min(freqs), np.max(freqs)
    plt.hist(freqs); plt.xlabel('hz'); plt.legend(['mean {:.1f} std {:.1f}\n min {:.1f} max {:.1f}'.format(_mean, _std, _min, _max)])
    plt.show()
    
if __name__ == '__main__':
    robot_pierrette, robot_caroline = range(2)
    robot = robot_caroline
    if robot == robot_pierrette:
        intr_cam_calib_path = '/home/poine/.ros/camera_info/ueye_drone.yaml'
        extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/pierrette_cam1_extr.yaml'
    elif robot == robot_caroline:
        intr_cam_calib_path = '/home/poine/.ros/camera_info/camera_road_front.yaml'
        extr_cam_calib_path = '/home/poine/work/roverboard/roverboard_description/cfg/caroline_cam_road_front_extr.yaml'
    cam = flvu.load_cam_from_files(intr_cam_calib_path, extr_cam_calib_path)

    pipe_1, pipe_2, pipe_3 = range(3)
    pipe_type = pipe_1
    if pipe_type == pipe_1:
        pipe = flvu.Contour1Pipeline(cam)
        pipe.thresholder.set_threshold(110)
    elif pipe_type == pipe_2:
        pipe = flvu.Contour2Pipeline(cam, flvu.CarolineBirdEyeParam());
        pipe.display_mode = flvu.Contour2Pipeline.show_be
        pipe.thresholder.set_threshold(110)
    elif pipe_type == pipe_3:
        pipe = flvu.Foo3Pipeline(cam, flvu.CarolineBirdEyeParam())
    mode_img, mode_bag = range(2)
    mode = mode_bag
    if mode == mode_img:
        #img_path = '/home/poine/work/robot_data/jeanmarie/z_room_line_11.png'
        img_path = '/home/poine/work/robot_data/caroline/line_z_02.png'
        test_on_img(pipe, cam, cv2.imread(img_path, cv2.IMREAD_COLOR))
    elif mode == mode_bag:
        bag_path = '/home/poine/2019-07-08-16-37-38.bag' # pierrette
        bag_path = '/home/poine/2019-07-15-19-01-30.bag'#'/home/poine/2019-07-11-18-08-11.bag' #2019-07-11-15-03-18.bag' # caroline
        img_topic = '/camera_road_front/image_raw'
        test_on_bag(pipe, cam, bag_path, img_topic)
