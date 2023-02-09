#! /usr/bin/env python

from tf import TransformBroadcaster
import rospy
from rospy import Time 

def main():
    rospy.init_node('my_broadcaster')
    
    b = TransformBroadcaster()
    
    translation = (0.38, 0.0, 0.25)
    rotation = (0.0, 0.0, 0.0, 1.0)
    rate = rospy.Rate(30)  # was 5hz
    
    while not rospy.is_shutdown():
        b.sendTransform(translation, rotation, Time.now(), 'base_link', 'camera_color_optical_frame')
        # b.sendTransform(translation, rotation, Time.now(), 'base_link', 'camera_imu_optical_frame')
        rate.sleep()
    


if __name__ == '__main__':
    main()