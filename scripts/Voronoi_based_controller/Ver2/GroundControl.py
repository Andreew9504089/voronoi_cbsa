#!/usr/bin/env python3

import rospy
from voronoi_cbsa.msg import TargetInfoArray, TargetInfo

import pygame
import numpy as np
from time import time, sleep
from ptz import PTZCamera
from math import cos, acos, sqrt, exp, sin
from pygame_recorder import ScreenRecorder
from scipy.stats import multivariate_normal
import itertools

initialized = False

class UAVs():
    def __init__(self, map_size, resolution):
        
        self.members = []
        self.map_size = map_size
        self.grid_size = resolution

    def AddMember(self, ptz_info):
        
        ptz = PTZCamera(ptz_info, self.map_size, self.grid_size)
        self.members.append(ptz)
        
        return

    # inefficient way, might come up with other data structure to manage the swarm 
    def DeleteMember(self, id): 
        
        for i in range(len(self.members)):
            if self.members.id == id:
                del self.members[i]
                break

        return

class Visualize():
    def __init__(self, map_size, grid_size):
        self.size = (np.array(map_size) / np.array(grid_size)).astype(np.int64)
        self.grid_size = grid_size
        self.window_size = np.array(self.size)*4
        self.display = pygame.display.set_mode(self.window_size)
        self.display.fill((0,0,0))
        self.blockSize = int(self.window_size[0]/self.size[0]) #Set the size of the grid block
        self.recorder = ScreenRecorder(1024, 1024, 60)

        for x in range(0, self.window_size[0], self.blockSize):
            for y in range(0, self.window_size[1], self.blockSize):
                rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                pygame.draw.rect(self.display, (125,125,125), rect, 1)

        pygame.display.update()

    def Visualize2D(self, cameras, global_voronoi, event_plt, targets, centroids, sub_centroids, sub_global_voronoi):

        map_plt = np.zeros((event_plt.shape)) - 1

        sum, sum1 = 0, 0
        for i in range(len(cameras)):
            map_plt = cameras[i].map_plt + map_plt
            sum += cameras[i].intercept_quality
            sum1 += cameras[i].coverage_quality

        print("Interception Quality: ",sum, "Coverage Quality: ",sum1)

        x_map = 0
        for x in range(0, self.window_size[0], self.blockSize):

            y_map = 0

            for y in range(0, self.window_size[1], self.blockSize):

                dense = event_plt[x_map][y_map]
                w = 0.6

                id = int(map_plt[y_map][x_map])

                if id not in range(len(cameras)):
                    color = [0,0,0]
                    # color[0] = (1-w)*(50+cameras[int(global_voronoi[x_map][y_map])].color[0]) + w*dense
                    # color[1] = (1-w)*(50+cameras[int(global_voronoi[x_map][y_map])].color[1]) + w*dense
                    # color[2] = (1-w)*(50+cameras[int(global_voronoi[x_map][y_map])].color[2]) + w*dense
                    color[0] = (1-w)*(50+cameras[int(sub_global_voronoi[x_map][y_map])].color[0]) + w*dense
                    color[1] = (1-w)*(50+cameras[int(sub_global_voronoi[x_map][y_map])].color[1]) + w*dense
                    color[2] = (1-w)*(50+cameras[int(sub_global_voronoi[x_map][y_map])].color[2]) + w*dense
                    rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                    pygame.draw.rect(self.display, color, rect, 0)
                else:
                    color = ((1-w)*cameras[id].color[0] + w*dense, \
                                (1-w)*cameras[id].color[1] + w*dense,\
                                    (1-w)*cameras[id].color[2] + w*dense)
                    rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                    pygame.draw.rect(self.display, color, rect, 0)

                y_map += 1
            x_map += 1 

        for camera in cameras:
            color = (camera.color[0], camera.color[1], camera.color[2])
            center = camera.last_pos/self.grid_size*self.blockSize
            font = pygame.font.Font('freesansbold.ttf', 16)
            text = font.render(camera.role, True, color)
            textRect = text.get_rect()
            textRect.center = (center[0], center[1] - 20) if camera.perspective[1] > 0 \
                                else (center[0], center[1] + 20) 
            R = camera.R*cos(camera.alpha)/self.grid_size[0]*self.blockSize
            self.display.blit(text, textRect)
            pygame.draw.line(self.display, color, center, center + camera.perspective*R, 3)
            pygame.draw.circle(self.display, color, camera.last_pos/self.grid_size*self.blockSize, 10)
            pygame.draw.circle(self.display, color, camera.last_pos/self.grid_size*self.blockSize,
                                camera.max_speed*self.blockSize/self.grid_size[0], 1)

            camera.last_pos = camera.pos

        for i in range(len(cameras)):

            if cameras[i].role == "Interceptor":
                centroid = (sub_centroids[i][0]*self.blockSize, sub_centroids[i][1]*self.blockSize)
                p1 = (centroid[0] + 10, centroid[1] + 10)
                p2 = (centroid[0] - 10, centroid[1] + 10)
                p3 = (centroid[0], centroid[1] - 10)
                pygame.draw.polygon(self.display, cameras[i].color,(p1, p2, p3))

            else:
                centroid = (centroids[i][0]*self.blockSize, centroids[i][1]*self.blockSize)
                p1 = (centroid[0] + 10, centroid[1] + 10)
                p2 = (centroid[0] - 10, centroid[1] + 10)
                p3 = (centroid[0], centroid[1] - 10)
                pygame.draw.polygon(self.display, cameras[i].color,(p1, p2, p3))
             
        for target in targets:

            pos = np.asarray(target[0])/self.grid_size*self.blockSize
            pygame.draw.circle(self.display, (0,0,0), pos, 4)
            pygame.draw.line(self.display, (0,0,0), pos, pos + target[3].reshape(1, 2)[0]/2\
                                /self.grid_size*self.blockSize, 2)
        
        pygame.draw.rect(self.display, (0, 0, 0), (0, 0, map_size[0]/grid_size[0]*self.blockSize, \
                                                    map_size[1]/grid_size[1]*self.blockSize), width = 3)
        
        pygame.display.flip()
        self.recorder.capture_frame(self.display)

def ComputeGlobalVoronoi(PTZs, event):
    global initialized

    ### Compute Voronoi Diagram for Tracker and Interceptor ###
    # create 2D arrays of x and y coordinates
    x_coords, y_coords = np.meshgrid(np.arange(event.shape[0]), np.arange(event.shape[1]), indexing='ij')

    for i in range(len(PTZs)):

        pos_self = PTZs[i].pos
        grid_size = PTZs[i].grid_size

        #r = PTZs[i].R*cos(PTZs[i].alpha)/grid_size[0]
        #r_sub = PTZs[i].max_speed/grid_size[0]

        #cost = ((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2 - r**2)*event
        cost = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*event
        if i == 0 and PTZs[i].role == "Tracker":
            cost_map = np.expand_dims(cost, axis = 0)

        elif i == 0:
            cost_map = np.expand_dims(np.full((event.shape), np.inf), axis=0)

        elif PTZs[i].role == "Tracker":
            cost_map = np.concatenate([cost_map, np.expand_dims(cost, axis = 0)])

        else:
            cost_map = np.concatenate([cost_map, np.expand_dims(np.full((event.shape), np.inf), axis = 0)])

        # cost_sub = ((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2 - r_sub**2)*event
        cost_sub = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*event
        if i == 0:
            cost_map_sub = np.expand_dims(cost_sub, axis = 0)

        else:
            cost_map_sub = np.concatenate([cost_map_sub, np.expand_dims(cost_sub, axis = 0)])

    global_voronoi = np.argmin(cost_map, axis = 0)
    sub_global_voronoi = np.argmin(cost_map_sub, axis = 0)

    centroids = [(0,0) for i in range(len(PTZs))]
    sub_centroids = [(0,0) for i in range(len(PTZs))]
    geo_centers = [(0,0) for i in range(len(PTZs))]

    for i in range(len(PTZs)):

        if PTZs[i].role == "Tracker":
            indices = np.where(global_voronoi == i)
            weighted_x = int(np.average(indices[0], weights=event[indices]))
            weighted_y = int(np.average(indices[1], weights=event[indices]))
            centroids[i] = ((weighted_x, weighted_y))
            geo_centers[i] = ((int(np.mean(indices[0])), int(np.mean(indices[1]))))
        
        indices_sub = np.where(sub_global_voronoi == i)
        weighted_x_sub = int(np.average(indices_sub[0], weights=event[indices_sub]))
        weighted_y_sub = int(np.average(indices_sub[1], weights=event[indices_sub]))
        sub_centroids[i] = ((weighted_x_sub, weighted_y_sub))
    
    initialized = True
    return global_voronoi, centroids, geo_centers, sub_centroids, sub_global_voronoi

def norm(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]**2

    return sqrt(sum)

def ComputeEventDensity(map_size, target, grid_size):

    event = []

    for i in range(len(target)):

        x, y = np.mgrid[0:map_size[0]:grid_size[0], 0:map_size[1]:grid_size[1]]
        xy = np.column_stack([x.flat, y.flat])
        mu = np.array(target[i][0])
        sigma = np.array([target[i][1], target[i][1]])
        covariance = np.diag(sigma**2)
        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
        event.append(z.reshape(x.shape))
    
    combinations = [list(itertools.combinations(range(len(target)), i)) for i in range(1, len(target)+1)]
    union = np.zeros(event[0].shape)

    for c in combinations:
    
        for pair in c:

            inter = np.ones(event[0].shape)

            for i in pair:

                inter = np.multiply(inter, event[i][:,:])

            union += ((-1)**(len(pair) + 1))*inter
        
    return union

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
    Test = False

    if Test:
        pygame.init()

        map_size = np.array([24, 24])    
        grid_size = np.array([0.1, 0.1])

        cameras = []
        camera0 = { 'id'            :  0,
                    'position'      :  np.array([6.,3.]),
                    'perspective'   :  np.array([0.9,1]),
                    'AngleofView'   :  10,
                    'range_limit'   :  2,
                    'lambda'        :  4,
                    'color'         : (200, 0, 0),
                    'max_speed'     :  1}

        cameras.append(camera0)

        camera1 = { 'id'            :  1,
                    'position'      :  np.array([3.,4.]),
                    'perspective'   :  np.array([0.7,1]),
                    'AngleofView'   :  10,
                    'range_limit'   :  2,
                    'lambda'        :  4,
                    'color'         : (0, 200, 0),
                    'max_speed'     :  3}

        cameras.append(camera1)

        camera2 = { 'id'            :  2,
                    'position'      :  np.array([4.,5.]),
                    'perspective'   :  np.array([0.7,1]),
                    'AngleofView'   :  10,
                    'range_limit'   :  2,
                    'lambda'        :  4,
                    'color'         : (50, 50, 200),
                    'max_speed'     :  1}

        cameras.append(camera2)

        # camera3 = { 'id'            :  3,
        #             'position'      :  np.array([7.4,5.]),
        #             'perspective'   :  np.array([0.7,1]),
        #             'AngleofView'   :  10,
        #             'range_limit'   :  2,
        #             'lambda'        :  4,
        #             'color'         : (150, 100, 200),
        #             'max_speed'     :  1}

        # cameras.append(camera3)
        
        # camera4 = { 'id'            :  4,
        #             'position'      :  np.array([7.1,5.]),
        #             'perspective'   :  np.array([0.7,1]),
        #             'AngleofView'   :  10,
        #             'range_limit'   :  2,
        #             'lambda'        :  4,
        #             'color'         : (100, 150, 200),
        #             'max_speed'     :  1}

        # cameras.append(camera4)

        # camera5 = { 'id'            :  5,
        #             'position'      :  np.array([7.4,5.6]),
        #             'perspective'   :  np.array([0.7,1]),
        #             'AngleofView'   :  10,
        #             'range_limit'   :  2,
        #             'lambda'        :  4,
        #             'color'         : (100, 150, 75),
        #             'max_speed'     :  1}

        # cameras.append(camera5)

        # camera6 = { 'id'            :  6,
        #             'position'      :  np.array([6.4,5.]),
        #             'perspective'   :  np.array([0.7,1]),
        #             'AngleofView'   :  10,
        #             'range_limit'   :  2,
        #             'lambda'        :  2,
        #             'color'         : (200, 100, 200),
        #             'max_speed'     :  1}

        # cameras.append(camera6)


        # Initialize UAV team with PTZ cameras
        uav_team = UAVs(map_size, grid_size)

        for camera in cameras:
            uav_team.AddMember(camera)

        # initialize environment with targets
        size = (map_size/grid_size).astype(np.int64)
        event = np.zeros((size[0], size[1]))

        # target's [position, standard diviation, weight, velocity]
        targets = [[(7, 18), 0.5, 10, RandomUnitVector()], \
                    [(12, 11), 0.5, 10,RandomUnitVector()]]
        event1 = ComputeEventDensity(map_size, targets, grid_size)

        # Start Simulation
        Done = False
        vis = Visualize(map_size, grid_size)

        while not Done:

            for op in pygame.event.get():
                if op.type == pygame.QUIT:
                    Done = True 

            for i in range(len(targets)):
                pos, vel = TargetDynamics(targets[i][0][0], targets[i][0][1], targets[i][3], grid_size[0])
                targets[i][0] = pos
                targets[i][3] = vel

            event1 = ComputeEventDensity(map_size, targets, grid_size)
            event_plt1 = ((event1 - event1.min()) * (1/(event1.max() - event1.min()) * 255)).astype('uint8')
            # Update Global and Local Voronoi 
            global_voronoi, centroids, geo_centers, sub_centroids, sub_global_voronoi = \
                ComputeGlobalVoronoi(uav_team.members, event1)

            for i in range(len(uav_team.members)):

                neighbors = [uav_team.members[j] for j in range(len(uav_team.members)) if j != i]

                if uav_team.members[i].role == "Tracker":
                    uav_team.members[i].Update(targets, neighbors, centroids[i], geo_centers[i], sub_global_voronoi)
                elif uav_team.members[i].role == "Interceptor":
                    uav_team.members[i].Update(targets, neighbors, sub_centroids[i], geo_centers[i], sub_global_voronoi)

            vis.Visualize2D(uav_team.members, global_voronoi, event_plt1, targets, centroids, \
                                sub_centroids, sub_global_voronoi)

        pygame.quit()

    else:
        rospy.init_node('ground_control_system', anonymous=True, disable_signals=True)
        rate = rospy.Rate(60)

        target_pub = rospy.Publisher("/target", TargetInfoArray, queue_size=10)
        
        def random_pos(pos=(0,0)):
            if pos == (0,0):
                x = np.random.random()*15 + 5
                y = np.random.random()*15 + 5
                return np.array((x,y))
            else:
                return np.asarray(pos)
        
        targets = [[random_pos((12,8)), 0.5, 10,RandomUnitVector()],
                   [random_pos((20,18)), 0.5, 10,RandomUnitVector()],
                   [random_pos((4,18)), 0.5, 10,RandomUnitVector()]]
        
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

                tmp.append(target_msg)

            targets_array = TargetInfoArray()
            targets_array.targets = [tmp[i] for i in range(len(tmp))]

            target_pub.publish(targets_array)
            rate.sleep()
            

          

