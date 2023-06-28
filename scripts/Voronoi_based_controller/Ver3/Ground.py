#!/usr/bin/env python3

import rospy
from voronoi_cbsa.msg import TargetInfoArray, TargetInfo

import pygame
import numpy as np
from time import time, sleep
from control import PTZCamera
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal
import itertools


def norm(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]**2

    return sqrt(sum)

def TargetDynamics(x, y, v, res):
    turn = np.random.randint(-30, 30)/180*np.pi
    rot = np.array([[cos(turn), -sin(turn)],
                    [sin(turn), cos(turn)]])
    v = rot@v.reshape(2,1)
    vx = v[0] if v[0]*res + x > 0 and v[0]*res + x < 24 else -v[0]
    vy = v[1] if v[1]*res + y > 0 and v[1]*res + y < 24 else -v[1]

    return (x,y), np.asarray([[0],[0]])
    #return (np.round(float(np.clip(v[0]*res + x, 0, 24)),1), np.round(float(np.clip(v[1]*res + y, 0, 24)),1)),\
    #            np.round(np.array([[vx],[vy]]), len(str(res).split(".")[1]))

def RandomUnitVector():
    v = np.asarray([np.random.normal() for i in range(2)])
    return v/norm(v)

if __name__ == "__main__":
    rospy.init_node('ground_control_station', anonymous=True, disable_signals=True)
    rate = rospy.Rate(60)

    target_pub = rospy.Publisher("/target", TargetInfoArray, queue_size=10)
    
    def random_pos(pos=(0,0)):
        if pos == (0,0):
            x = np.random.random()*15 + 5
            y = np.random.random()*15 + 5
            return np.array((x,y))
        else:
            return np.asarray(pos)
    
    targets = [[random_pos((12,8)), 0.5, 10,RandomUnitVector(), ['camera', 'smoke_detector']],
                [random_pos((20,18)), 0.5, 10,RandomUnitVector(), ['camera', 'manipulator']],
                [random_pos((4,18)), 0.5, 10,RandomUnitVector()], ['camera', 'smoke_detector', 'manipulator']]
    
    while not rospy.is_shutdown():
            
        grid_size = rospy.get_param("/grid_size", 0.1)
        tmp = []

        for i in range(len(targets)):
            pos, vel = TargetDynamics(targets[i][0][0], targets[i][0][1], targets[i][3], grid_size)
            targets[i][0] = pos
            targets[i][3] = vel

            target_msg = TargetInfo()
            target_msg.id = i
            target_msg.position.x = pos[0]
            target_msg.position.y = pos[1]
            target_msg.standard_deviation = targets[i][1]
            target_msg.weight = targets[i][2]
            target_msg.velocity.linear.x = vel[0]
            target_msg.velocity.linear.y = vel[1]
            target_msg.required_sensor = targets[i][4]

            tmp.append(target_msg)

        targets_array = TargetInfoArray()
        targets_array.targets = [tmp[i] for i in range(len(tmp))]

        target_pub.publish(targets_array)
        rate.sleep()