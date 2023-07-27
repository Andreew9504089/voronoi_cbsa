#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, Point
from voronoi_cbsa.msg import ExchangeData, ExchangeDataArray, TargetInfoArray, SensorArray, Sensor, ValidSensors, WeightArray, Weight
from std_msgs.msg import Int16, Float32MultiArray, Int16MultiArray, Float32, Float64, Float64MultiArray, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import pandas as pd
from time import time, sleep
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import itertools
import math
import _thread


class PTZCamera():
    def __init__(self, map_size, grid_size, general_properties,
                 coop, balance, strength, camera_properties=None, 
                 manipulator_properties=None, smoke_detector_properties=None, K_p=0.8, K_v=0.5, step = 0.1):

        self.K_p            = K_p
        self.K_v            = K_v
        self.step           = step
        self.cooperation    = coop 
        self.sensor_balance = balance
        self.w_coop         = strength
        self.safe_distance  = 1
        self.avoid_weight   = 0.05
        
        # Setting up environment parameters
        self.total_agents       = rospy.get_param('/total_agents', 1)
        self.visualize_option   = rospy.get_param('/visualize_option', default = -1)
        self.grid_size          = grid_size
        self.map_size           = map_size
        self.size               = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
        
        # Setting up agent's general parameters
        self.id                 = general_properties['id']
        self.pos                = general_properties['position']
        self.valid_sensors      = general_properties['valid_sensor']
        self.max_speed          = general_properties['max_speed']
        self.sensor_qualities   = {}

        if self.valid_sensors['camera']:
            # Camera property
            self.perspective        = camera_properties['perspective']/self.Norm(camera_properties['perspective'])
            self.camera_range       = camera_properties['desired range']
            self.angle_of_view      = camera_properties['angle of view']
            self.camera_variance    = camera_properties['variance']
        else:
            # Camera property
            self.perspective        = np.array([1.,0.])
            self.camera_range       = 0
            self.angle_of_view      = 0
            self.camera_variance    = 0
            
        if self.valid_sensors['manipulator']:
            # Manipulator property
            self.operation_range    = manipulator_properties['arm length']
            self.approx_param       = manipulator_properties['k']
        else:
            # Manipulator property
            self.operation_range    = 0
            self.approx_param       = 0
            
        if self.valid_sensors['smoke_detector']:
            # Smoke Detector property
            self.smoke_variance = smoke_detector_properties['variance']
        else:
            self.smoke_variance = 0
            
        self.target_ready        = False
        self.targets                = {}
        self.target_buffer          = {}
            
        self.event_density          = {}
        self.event_density_buffer   = {}
        
        self.neighbors              = []
        self.neighbors_buffer       = {}
        
        self.sensor_graph           = {}
        self.sensor_voronoi         = {}
        self.sensor_weight          = {}
        
        self.total_score = 0
        for sensor in self.valid_sensors.keys():
            self.sensor_graph[sensor]       = []
            self.sensor_voronoi[sensor]     = {}  
            self.sensor_qualities[sensor]   = {}  
            self.sensor_weight[sensor]      = {} 
            
        self.RosInit()
        self.debug = self.DEBUGTOOL(id = self.id)
        
    def RosInit(self):

        rospy.Subscriber("local/neighbor_info", ExchangeDataArray, self.NeighborCallback)
        rospy.Subscriber("local/target", TargetInfoArray, self.TargetCallback)        
        
        self.pub_pos                = rospy.Publisher("local/position", Point, queue_size=10)
        self.pub_exchange_data      = rospy.Publisher("local/exchange_data",ExchangeData, queue_size=10)
        
        self.pub_sensor_weight      = rospy.Publisher("visualize/sensor_weights", WeightArray, queue_size=10)
        self.pub_total_score        = rospy.Publisher("visualize/total_score", Float64, queue_size=10)
        self.pub_sensor_scores      = rospy.Publisher("visualize/sensor_scores", SensorArray, queue_size=10)
        self.pub_pose               = rospy.Publisher("visualize/pose", Pose, queue_size=10)
        self.pub_valid_sensors      = rospy.Publisher("visualize/valid_sensors", ValidSensors, queue_size=10)
        
        self.pub_sensor_graphs      = {}
        
        for sensor in self.valid_sensors.keys():
            if self.valid_sensors[sensor]:
                self.pub_sensor_graphs[sensor] = rospy.Publisher("visualize/"+sensor+"_neighbor", Int16MultiArray, queue_size = 10)
        
    def NeighborCallback(self, msg):
        self.neighbors_buffer = {}
        
        for neighbor in msg.data:

            pos_x   = neighbor.position.x
            pos_y   = neighbor.position.y
            pos     = np.array([pos_x, pos_y])
            role    = {}
            weights = {}
            sensor_scores = {}
            
            for sensor in neighbor.role.sensors:
                role[sensor.type] = sensor.score

            for weight in neighbor.weights.weights:
                weights[weight.type] = {}
                
            for weight in neighbor.weights.weights:
                weights[weight.type][weight.event_id] = weight.score
            
            for score in neighbor.sensor_scores.weights:
                sensor_scores[score.type] = {}
            
            for score in neighbor.sensor_scores.weights:
                sensor_scores[score.type][score.event_id] = score.score
                    
            self.neighbors_buffer[neighbor.id] = {"position":   pos, "role": role, "operation_range": neighbor.operation_range,
                                                  "approx_param": neighbor.approx_param, "smoke_variance": neighbor.smoke_variance,
                                                  "camera_range": neighbor.camera_range, "angle_of_view": neighbor.angle_of_view,
                                                  "camera_variance": neighbor.camera_variance, "weights": weights, "sensor_scores": sensor_scores}
    
    def TargetCallback(self, msg):
        self.target_ready = True
        for target in msg.targets:

            pos_x   = target.position.x
            pos_y   = target.position.y
            pos     = np.array([pos_x, pos_y])

            std     = target.standard_deviation
            weight  = target.weight

            vel_x   = target.velocity.linear.x
            vel_y   = target.velocity.linear.y
            vel     = np.array([vel_x, vel_y])
            
            requirements = [target.required_sensor[i] for i in range(len(target.required_sensor))]

            self.target_buffer[target.id] = [pos, std, weight, vel, target.id, requirements]
                
         
    def PublishInfo(self):
        # publish position
        pos     = Point()
        pos.x   = self.pos[0]
        pos.y   = self.pos[1]
        self.pub_pos.publish(pos)

        # Publish total score
        total_score  = Float64()

        total_score.data = self.total_score
        self.pub_total_score.publish(total_score)
        
        # # publish sensor scores
        # sensor_scores = SensorArray()
        # sensor_scores.sensors = []
        # for i in self.valid_sensors.keys():
        #     if self.valid_sensors[i]:
        #         sensor = Sensor()
        #         sensor.type = i
        #         print(self.sensor_scores[i])
        #         sensor.score = self.sensor_scores[i]
        #         sensor_scores.sensors.append(sensor)
                
        # self.pub_sensor_scores.publish(sensor_scores)
                
        #publish exchange data
        data = ExchangeData()

        data.id = self.id
        data.position.x = self.pos[0]
        data.position.y = self.pos[1]
        data.operation_range = self.operation_range
        
        sensor_arr = SensorArray()
        sensor_arr.sensors = []
        weight_arr = WeightArray()
        weight_arr.weights = []
        scores_arr = WeightArray()
        scores_arr.weights = []
        
        for i in self.valid_sensors.keys():
            sensor = Sensor()
            sensor.type = i
            sensor.score = self.valid_sensors[i]
            sensor_arr.sensors.append(sensor)

            for event in self.targets.keys():
                if event in self.sensor_weight[i].keys():
                    weight          = Weight()
                    weight.event_id = event
                    weight.type     = i
                    weight.score    = self.sensor_weight[i][event]
                    weight_arr.weights.append(weight)
                
                else:
                    weight          = Weight()
                    weight.event_id = event
                    weight.type     = i
                    weight.score    = 0
                    weight_arr.weights.append(weight)
                
                weight          = Weight()
                weight.event_id = event
                weight.type     = i
                weight.score    = self.sensor_scores[i][event]
                scores_arr.weights.append(weight)
                
                
        data.operation_range    = self.operation_range
        data.approx_param       = self.approx_param
        data.smoke_variance     = self.smoke_variance
        data.camera_range       = self.camera_range
        data.angle_of_view      = self.angle_of_view
        data.camera_variance    = self.camera_variance
        
        data.sensor_scores  = scores_arr
        data.weights        = weight_arr
        data.role           = sensor_arr
   
        self.pub_exchange_data.publish(data)
        self.pub_sensor_weight.publish(weight_arr)
        
        # publish valid sensors
        valid_sensor = ValidSensors()
        valid_sensor.data = []
        for sensor in self.valid_sensors.keys():
            if self.valid_sensors[sensor]:
                valid_sensor.data.append(sensor)
        self.pub_valid_sensors.publish(valid_sensor)
        
        # publish pose
        pose = Pose()
        pose.position.x = self.pos[0]
        pose.position.y = self.pos[1]
        pose.orientation.x = self.perspective[0]
        pose.orientation.y = self.perspective[1]        
        
        self.pub_pose.publish(pose)
        
        # publish sensor graphs
        for sensor in self.valid_sensors.keys():
            if self.valid_sensors[sensor] and sensor in self.neighbors.keys():
                neighbor = [i for i in self.neighbors[sensor]]
                self.pub_sensor_graphs[sensor].publish(neighbor)
        
    # Find the agent's voronoi neighbors
    def ComputeNeighbors(self, role):
        def CountNeighborRole(neighbors, role):
            neighbor = []
            if role == -1:
                for n in neighbors.keys():
                    neighbor.append(n)
            
            else:
                for n in neighbors.keys():
                    if neighbors[n]["role"][role] > 0:
                        neighbor.append(n)

            return neighbor
        
        valid_neighbor = CountNeighborRole(self.neighbors, role)
        
        if len(valid_neighbor) >= 4:

            keys = [self.id]
            for key in valid_neighbor:
                keys.append(key)

            idx_map = {}
            idx_list = []

            for i, key in enumerate(keys):
                idx_map[key] = i
                idx_list.append(key)

            points = [0 for i in keys]
            points[0] = self.pos/self.grid_size

            for member in valid_neighbor:
                pos = np.asarray([self.neighbors[member]["position"][0], self.neighbors[member]["position"][1]])
                points[idx_map[member]] = pos/self.grid_size

            points = np.asarray(points)
            tri = Delaunay(points)

            ids = []
            for simplex in tri.simplices:
                if idx_map[self.id] in simplex:
                    for id in simplex:
                        ids.append(id)

            neighbors = []
            for member in valid_neighbor:
                if idx_map[member] in ids:

                    neighbors.append(member)
            
        elif len(valid_neighbor) >= 0:

            neighbors = []
            for member in valid_neighbor:
                neighbors.append(member)
            
        else:
            neighbors = []            
               
        return neighbors
                        
    def Update(self):
        if self.target_ready:
            self.neighbors          = self.neighbors_buffer.copy()                      
            self.targets            = self.target_buffer.copy()   
                
            self.total_score    = 0
            self.sensor_scores = {}
            for i in self.valid_sensors.keys():
                self.sensor_scores[i] = {}
                for event in self.targets.keys():
                    self.sensor_scores[i][event]  = 0.
            
            for target in self.targets.keys():
                self.event_density[target]    = self.ComputeEventDensity(target = self.targets[target])
            
            for sensor in self.valid_sensors.keys():
                self.sensor_graph[sensor]       = []
                self.sensor_voronoi[sensor]     = {}   
                self.sensor_qualities[sensor]   = {}  
                self.sensor_weight[sensor]      = {}       
            
            self.ComputeSensorWeight()
            
            self.sensor_graph[-1] = self.ComputeNeighbors(role = -1)
            
            for role in self.valid_sensors.keys():
                self.sensor_graph[role] = self.ComputeNeighbors(role=role)
                for event in self.targets.keys():
                    self.sensor_voronoi[role][event] = np.zeros(self.size, dtype=object)
                    self.UpdateSensorVoronoi(role = role, event = event)
                        
            u_p, u_v = self.ComputeControlSignal()
            self.UpdatePosition(u_p)
            
            if self.valid_sensors['camera']:
                self.UpdatePerspective(u_v)
            
            self.ComputeUtility()    
            self.PublishInfo()
            
   
    def UpdatePosition(self, u_p):
        u_p = self.max_speed*(u_p/np.linalg.norm(u_p))

        
        u_avoid = np.array([0., 0.])
        # for neighbor in self.sensor_graph[-1]:
        #     neighbor_pos = self.neighbors[neighbor]['position']
        #     vec = self.pos - neighbor_pos
        #     u_avoid -= (self.safe_distance - np.linalg.norm(vec))*(vec/np.linalg.norm(vec))
        
        # u_avoid /= self.max_speed*np.linalg.norm(u_avoid) 
          
        self.pos += self.K_p * ((1 - self.avoid_weight)*u_p + self.avoid_weight*u_avoid) * self.step if not np.isnan(u_p)[0] else self.pos
        
        if self.pos[0] < 0:
            self.pos[0] = 0
        elif self.pos[0] > self.map_size[0]:
            self.pos[0] = self.map_size[0]
            
        if self.pos[1] < 0:
            self.pos[1] = 0
        elif self.pos[1] > self.map_size[1]:
            self.pos[1] = self.map_size[1]
                
    def UpdatePerspective(self, u_v):
        
        #u_v /= np.linalg.norm(u_v)
        try:
            turning = math.acos((u_v @ self.perspective.T)/np.linalg.norm(u_v))
            if  turning > 15/180*np.pi:
                u_v *= (15/180*np.pi)/turning
        except:
            u_v = np.array([0., 0.])
            
        self.perspective += self.K_v*u_v*self.step
        self.perspective /= self.Norm(self.perspective)

        if self.pos[0] + self.perspective[0] < 0 or self.pos[0] + self.perspective[0] > 24:
           self.perspective[0] *= -1  
        
        if self.pos[1] + self.perspective[1] < 0 or self.pos[1] + self.perspective[1] > 24:
           self.perspective[1] *= -1 
           
    def UpdateSensorVoronoi(self, role, event):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')

        global_event = np.zeros(self.size)
        target = self.targets[event]
        if role in target[5]:
            global_event = self.event_density[event]
            
            if self.valid_sensors[role]:
                pos_self = self.pos
                grid_size = self.grid_size
                
                total_cost = (((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2) - (100*self.sensor_weight[role][event])**2)*global_event
                sensor_voronoi = np.full(self.size, self.id, dtype=object)
                
            else:
                total_cost = np.full(self.size, np.inf, dtype=object)
                sensor_voronoi = np.full(self.size, -1)
                
            for neighbor in self.sensor_graph[role]:
                pos_self = self.neighbors[neighbor]["position"]
                grid_size = self.grid_size

                cost = (((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2) - (100*self.neighbors[neighbor]["weights"][role][event])**2)*global_event#self.ComputeCost(role, pos_self, grid_size, x_coords, y_coords)
                sensor_voronoi = np.where(cost < total_cost, neighbor, sensor_voronoi)
                total_cost = np.where(cost < total_cost, cost, total_cost)

            self.sensor_voronoi[role][event] = sensor_voronoi
    
    def ComputeSensorWeight(self):
        if self.sensor_balance:
            for event in self.targets.keys():
                for sensor in self.valid_sensors.keys():
                    single_sensor_sum = 0
                    if self.valid_sensors[sensor] and sensor in self.targets[event][5]:
                        for neighbor in self.neighbors.values():  
                            if neighbor['role'][sensor] > 0:
                                single_sensor_sum += neighbor['sensor_scores'][sensor][event]
                                
                        self.sensor_weight[sensor][event] = (1 + self.sensor_scores[sensor][event])/(1 + self.sensor_scores[sensor][event] + single_sensor_sum)
                    
                    else: 
                        self.sensor_weight[sensor][event] = 0

            total = 0
            for event in self.targets.keys():
                for sensor in self.valid_sensors.keys():
                    if self.valid_sensors[sensor]:
                        total += self.sensor_weight[sensor][event]
            
            for event in self.targets.keys():
                for sensor in self.valid_sensors.keys():
                    if self.valid_sensors[sensor]:
                        self.sensor_weight[sensor][event] = self.sensor_weight[sensor][event]/total if total > 0 else 1
            
        else:
            cnt = 0
            for event in self.targets.keys():
                for sensor in self.valid_sensors.keys():
                    if self.valid_sensors[sensor] and sensor in self.targets[event][5]:
                        cnt += 1
            
            for event in self.targets.keys():
                for sensor in self.valid_sensors.keys():
                    if self.valid_sensors[sensor] and sensor in self.targets[event][5]:
                        self.sensor_weight[sensor][event] = 1/cnt
                    else:
                        self.sensor_weight[sensor][event] = 0
        
        # if self.id == 2:
        #     print(self.sensor_weight)
                            
    def ComputeUtility(self):
        
        for event in self.targets.keys():
            for role in self.valid_sensors.keys():
                tmp = np.zeros(self.size, dtype=np.float64)
                quality = np.zeros(self.size, dtype=np.float64)
                if self.valid_sensors[role] and role in self.targets[event][5]:
                    quality = self.ComputeSelfQuality(role=role, event=event)
                    tmp = quality
                    
                    for k in self.valid_sensors.keys():
                        if k in self.targets[event][5] and k != role:
                            coop_quality = self.ComputeCooperateQuality(coop_role = role, coop_quality = quality, role=k, event = event, eval = True)

                            tmp *= (1 + self.w_coop*coop_quality)

                    tmp *= self.event_density[event]*self.grid_size[0]

                    self.total_score += self.sensor_weight[role][event]*np.sum(tmp)
                self.sensor_scores[role][event] = np.sum(quality)
                    
    def ComputeControlSignal(self):
        u_p = np.array([0., 0.])  
        u_v = np.array([0., 0.])     
        total_gradient = [np.zeros(self.size), np.zeros(self.size)]
        sensor_gradient = [np.zeros(self.size), np.zeros(self.size)]
        
        for event in self.targets.keys():
            for role in self.valid_sensors.keys():
                total_gradient[0] = np.zeros(self.size)
                total_gradient[1] = np.zeros(self.size)
                
                if self.valid_sensors[role] and role in self.targets[event][5]:
                    
                    sensor_gradient[0] = self.ComputeGradient(role, event, 'x')
                    sensor_gradient[1] = self.ComputeGradient(role, event, 'y')
                    
                    # sensor_gradient[0] /= np.linalg.norm(np.array(np.sum(sensor_gradient[0]), np.sum(sensor_gradient[1])))
                    # sensor_gradient[1] /= np.linalg.norm(np.array(np.sum(sensor_gradient[0]), np.sum(sensor_gradient[1])))
                    
                    perspective_gradient_x = self.ComputeGradient('camera', event, 'perspective_x')
                    perspective_gradient_y = self.ComputeGradient('camera', event, 'perspective_y')
                    
                    if self.cooperation == True:
                        for k in self.valid_sensors.keys():
                            if k in self.targets[event][5] and k != role:
                                coop_quality = self.ComputeCooperateQuality(coop_role = role, role=k, event = event)
                                sensor_gradient[0] *= (1 + self.w_coop*coop_quality)
                                sensor_gradient[1] *= (1 + self.w_coop*coop_quality)

                                if role == 'camera':
                                    perspective_gradient_x *= (1 + self.w_coop*coop_quality)
                                    perspective_gradient_y *= (1 + self.w_coop*coop_quality)
                    
                    if role == 'camera':
                        u_v[0] += self.sensor_weight[role][event]*np.sum(perspective_gradient_x * self.event_density[event])
                        u_v[1] += self.sensor_weight[role][event]*np.sum(perspective_gradient_y * self.event_density[event])                                        
                                
                    total_gradient[0] = sensor_gradient[0]
                    total_gradient[1] = sensor_gradient[1]
                    
                    if self.cooperation == True:
                        f = self.ComputeSelfQuality(role=role, event = event)
                        for i in self.valid_sensors.keys():
                            if self.valid_sensors[i] and i != role and k in self.targets[event][5]:
                                sensor_gradient[0] = self.w_coop*self.ComputeGradient(i, event, 'x')*(f)
                                sensor_gradient[1] = self.w_coop*self.ComputeGradient(i, event, 'y')*(f)
                                
                                for k in self.valid_sensors.keys():
                                    if k in self.targets[event][5] and k != i and k != role:
                                        coop_quality = self.ComputeCooperateQuality(coop_role = role, role=k, event = event).astype('int64')
                                        sensor_gradient[0] *= (1 + self.w_coop*coop_quality)
                                        sensor_gradient[1] *= (1 + self.w_coop*coop_quality)
                        
                                total_gradient[0] += sensor_gradient[0]
                                total_gradient[1] += sensor_gradient[1]

                total_gradient[0] *= self.event_density[event]
                total_gradient[1] *= self.event_density[event]
                
                tmp_x = self.sensor_weight[role][event]*np.sum(total_gradient[0])
                tmp_y = self.sensor_weight[role][event]*np.sum(total_gradient[1]) 


                u_p[0] = u_p[0] + tmp_x if not np.isnan(tmp_x) else u_p[0]
                u_p[1] = u_p[1] + tmp_y if not np.isnan(tmp_y) else u_p[1] 
        
        return u_p, u_v
              
    def ComputeGradient(self, role, event, type):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        pos_self = self.pos
        grid_size = self.grid_size
        gradient = np.zeros(self.size)
        
        if role == "camera":
            dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
            x = (x_coords*grid_size[0] - pos_self[0])
            y = (y_coords*grid_size[1] - pos_self[1])
            
            if type == 'x':
                gradient = ((pos_self[0] - x_coords*grid_size[0])*np.exp(-((dist - self.camera_range)**2)/(2*(self.camera_variance**2)))\
                            *(self.camera_range - dist)/((self.camera_variance**2)*dist))
                
                per_quality = (1/(1-np.cos(self.angle_of_view)))*((x*self.perspective[0] + y*self.perspective[1])/dist - np.cos(self.angle_of_view))
                gradient[per_quality < 0] = 0
                
                #gradient *= per_quality
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
                
            elif type == 'y':
                gradient = ((pos_self[1] - y_coords*grid_size[1])*np.exp(-((dist - self.camera_range)**2)/(2*(self.camera_variance**2)))\
                            *(self.camera_range - dist)/((self.camera_variance**2)*dist))
                
                per_quality = (1/(1-np.cos(self.angle_of_view)))*((x*self.perspective[0] + y*self.perspective[1])/dist - np.cos(self.angle_of_view))
                gradient[per_quality < 0] = 0
                
                # gradient *= per_quality
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
                
            elif type == 'perspective_x':    
                
                gradient = x/(dist)*(1-np.cos(self.angle_of_view))
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
            
            elif type == 'perspective_y':    
               
                gradient = y/(dist)*(1-np.cos(self.angle_of_view))
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
            
            gradient[np.isnan(gradient)] = 0
                
        elif role == "manipulator":
            dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
            if type == 'x':
                gradient = (2*self.approx_param*(x_coords*grid_size[0] - pos_self[0])*np.exp(-2*self.approx_param*(self.operation_range - dist)))\
                                /(((1+np.exp(-2*self.approx_param*(self.operation_range - dist)))**2)*dist)
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
            elif type == 'y':
                gradient = (2*self.approx_param*(y_coords*grid_size[1] - pos_self[1])*np.exp(-2*self.approx_param*(self.operation_range - dist)))\
                                /(((1+np.exp(-2*self.approx_param*(self.operation_range - dist)))**2)*dist)
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
            
            gradient[np.isnan(gradient)] = 0
            
        elif role == "smoke_detector":
            dist = (x_coords*grid_size[0] - pos_self[0])**2 + (y_coords*grid_size[1] - pos_self[1])**2
            
            if type == 'x':
                gradient = (x_coords*grid_size[0] - pos_self[0]) * np.exp(-(dist**2) / (2 * self.smoke_variance ** 2))
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
                
            elif type == 'y':
                gradient = (y_coords*grid_size[1] - pos_self[1]) * np.exp(-(dist**2) / (2 * self.smoke_variance ** 2))
                gradient = np.where(self.sensor_voronoi[role][event] == self.id, gradient, 0)
            
            gradient[np.isnan(gradient)] = 0
                
        return gradient
    
    def ComputeSelfQuality(self, role, event):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        pos_self = self.pos
        grid_size = self.grid_size

        self_territory = np.where(self.sensor_voronoi[role][event] == self.id, -1, self.sensor_voronoi[role][event])
        
        if role == 'manipulator':
            dist = self.operation_range - np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
            self_quality = 1/(1+np.exp(-2*self.approx_param*(dist)))
            self_quality = np.where(self_territory > -1, 0, self_quality)
        
        elif role == 'smoke_detector':
            dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
            self_quality = np.exp(-(dist**2)/(2*self.smoke_variance**2))
            self_quality = np.where(self_territory > -1, 0, self_quality)
            
        elif role == 'camera':
            dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)-self.camera_range
            self_quality = np.exp(-(dist**2)/(2*self.camera_variance**2))
            
            dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
            x = (x_coords*grid_size[0] - pos_self[0])
            y = (y_coords*grid_size[1] - pos_self[1])
            per_quality = (1/(1-np.cos(self.angle_of_view)))*((x*self.perspective[0] + y*self.perspective[1])/dist - np.cos(self.angle_of_view))
            self_quality[per_quality < 0] = 0
                    
            self_quality = np.where(self_territory > -1, 0, self_quality)
        
        return self_quality
    
    def ComputeCooperateQuality(self, coop_role, role, event = None, eval = False, coop_quality = None):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        pos_self = self.pos
        grid_size = self.grid_size

        if role == 'camera':
            
            total_quality = np.zeros(self.size)
            for neighbor in self.sensor_graph[role]+[self.id]:
                valid = self.valid_sensors[role] if neighbor == self.id else (self.neighbors[neighbor]["role"][role] > 0)
                
                if valid:
                    pos = self.neighbors[neighbor]["position"] if neighbor != self.id else self.pos
                    camera_range = self.camera_range if neighbor == self.id else self.neighbors[neighbor]["camera_range"]
                    camera_variance = self.camera_variance if neighbor == self.id else self.neighbors[neighbor]["camera_variance"]
                    dist = np.sqrt((pos[0] - x_coords*grid_size[0])**2 + (pos[1] - y_coords*grid_size[1])**2) - camera_range

                    territory = np.where(self.sensor_voronoi[role][event] == neighbor, -1, self.sensor_voronoi[role][event])
                    individual_quality = np.exp(-(dist**2)/(2*camera_variance**2))
                    
                    dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
                    x = (x_coords*grid_size[0] - pos_self[0])
                    y = (y_coords*grid_size[1] - pos_self[1])
                    per_quality = (1/(1-np.cos(self.angle_of_view)))*((x*self.perspective[0] + y*self.perspective[1])/dist - np.cos(self.angle_of_view))
                    individual_quality[per_quality < 0] = 0
                    
                    individual_quality = np.where(territory > -1, 0, individual_quality)
                    
                    total_quality += individual_quality
                    
                    if eval:               
                        #self_territory = np.where(self.sensor_voronoi[coop_role] == self.id, -1, self.sensor_voronoi[coop_role])
                        
                        if neighbor not in self.sensor_qualities[coop_role].keys():
                            self.sensor_qualities[coop_role][neighbor] = {}
                            self.sensor_qualities[coop_role][neighbor][event] = 0

                        elif event not in self.sensor_qualities[coop_role][neighbor].keys():
                            self.sensor_qualities[coop_role][neighbor][event] = 0
                            
                        # if coop_role == 'manipulator':
                        #     coop_dist = self.operation_range - np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[0])**2)
                        #     self_quality = 1/(1+np.exp(-2*self.approx_param*(coop_dist)))
                        #     self_quality = np.where(self_territory > -1, 0, self_quality)
                        #     self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                        
                        # elif coop_role == 'smoke_detector':
                        #     coop_dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[0])**2)
                        #     self_quality = np.exp(-(dist**2)/(2*self.smoke_variance**2))
                        #     self_quality = np.where(self_territory > -1, 0, self_quality)                    
                        #     self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                        
                        self.sensor_qualities[coop_role][neighbor][event] += self.w_coop*np.sum(individual_quality*coop_quality*self.event_density[event])
                        
        elif role == 'manipulator':
            total_quality = np.zeros(self.size)
            for neighbor in self.sensor_graph[role]+[self.id]:
                valid = self.valid_sensors[role] if neighbor == self.id else (self.neighbors[neighbor]["role"][role] > 0)
                
                if valid:
                    pos = self.neighbors[neighbor]["position"] if neighbor != self.id else self.pos
                    operation_range = self.operation_range if neighbor == self.id else self.neighbors[neighbor]['operation_range']
                    approx_param = self.approx_param if neighbor == self.id else self.neighbors[neighbor]['approx_param']
                    dist = operation_range - np.sqrt((pos[0] - x_coords*grid_size[0])**2 + (pos[1] - y_coords*grid_size[0])**2)

                    territory = np.where(self.sensor_voronoi[role][event] == neighbor, -1, self.sensor_voronoi[role][event])
                    individual_quality = 1/(1+np.exp(-2*approx_param*(dist)))
                    individual_quality = np.where(territory > -1, 0, individual_quality)
                    
                    total_quality += individual_quality
                    
                    if eval:           
                        #self_territory = np.where(self.sensor_voronoi[coop_role] == self.id, -1, self.sensor_voronoi[coop_role])

                        if neighbor not in self.sensor_qualities[coop_role].keys():
                            self.sensor_qualities[coop_role][neighbor] = {}
                            self.sensor_qualities[coop_role][neighbor][event] = 0
                            
                        elif event not in self.sensor_qualities[coop_role][neighbor].keys():
                            self.sensor_qualities[coop_role][neighbor][event] = 0
                            
                        # if coop_role == 'camera':
                        #     coop_dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[0])**2)-self.camera_range
                        #     self_quality = np.exp(-(dist**2)/(2*self.camera_variance**2))
                            
                        #     dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
                        #     x = (x_coords*grid_size[0] - pos_self[0])
                        #     y = (y_coords*grid_size[1] - pos_self[1])
                        #     per_quality = (1/(1-np.cos(self.angle_of_view)))*((x*self.perspective[0] + y*self.perspective[1])/dist - np.cos(self.angle_of_view))
                        #     self_quality[per_quality < 0] = 0
                    
                    
                        #     self_quality = np.where(self_territory > -1, 0, self_quality)
                        #     self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])

                        # elif coop_role == 'smoke_detector':
                        #     coop_dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[0])**2)
                        #     self_quality = np.exp(-(dist**2)/(2*self.smoke_variance**2))
                        #     self_quality = np.where(self_territory > -1, 0, self_quality)
                        #     self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                        
                        self.sensor_qualities[coop_role][neighbor][event] += self.w_coop*np.sum(individual_quality*coop_quality*self.event_density[event])
                        
        elif role == 'smoke_detector':
            total_quality = np.zeros(self.size, dtype=object)
            for neighbor in self.sensor_graph[role]+[self.id]:
                valid = self.valid_sensors[role] if neighbor == self.id else (self.neighbors[neighbor]["role"][role] > 0)
                
                if valid:
                    smoke_variance = self.smoke_variance if neighbor == self.id else self.neighbors[neighbor]['smoke_variance']
                    pos = self.neighbors[neighbor]["position"] if neighbor != self.id else self.pos
                    dist = np.sqrt((pos[0] - x_coords*grid_size[0])**2 + (pos[1] - y_coords*grid_size[0])**2)

                    territory = np.where(self.sensor_voronoi[role][event] == neighbor, -1, self.sensor_voronoi[role][event])
                    individual_quality = np.exp(-(dist**2)/(2*smoke_variance**2))
                    individual_quality = np.where(territory > -1, 0, individual_quality)
                    
                    total_quality += individual_quality
                    
                    if eval:          
                        #self_territory = np.where(self.sensor_voronoi[coop_role] == self.id, -1, self.sensor_voronoi[coop_role])
                        
                        if neighbor not in self.sensor_qualities[coop_role].keys():
                            self.sensor_qualities[coop_role][neighbor] = {}
                            self.sensor_qualities[coop_role][neighbor][event] = 0
                            
                        elif event not in self.sensor_qualities[coop_role][neighbor].keys():
                            self.sensor_qualities[coop_role][neighbor][event] = 0
                                      
                        # if coop_role == 'manipulator':
                        #     coop_dist = self.operation_range - np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[0])**2)
                        #     self_quality = 1/(1+np.exp(-2*self.approx_param*(coop_dist)))
                        #     self_quality = np.where(self_territory > -1, 0, self_quality)
                        #     self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                        
                        # elif coop_role == 'camera':
                        #     coop_dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[0])**2)-self.camera_range
                        #     self_quality = np.exp(-(dist**2)/(2*self.camera_variance**2))
                        #     dist = np.sqrt((pos_self[0] - x_coords*grid_size[0])**2 + (pos_self[1] - y_coords*grid_size[1])**2)
                        #     x = (x_coords*grid_size[0] - pos_self[0])
                        #     y = (y_coords*grid_size[1] - pos_self[1])
                        #     per_quality = (1/(1-np.cos(self.angle_of_view)))*((x*self.perspective[0] + y*self.perspective[1])/dist - np.cos(self.angle_of_view))
                        #     self_quality[per_quality < 0] = 0
                    
                        #     self_quality = np.where(self_territory > -1, 0, self_quality)
                        #     self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                        
                        self.sensor_qualities[coop_role][neighbor][event] += self.w_coop*np.sum(individual_quality*coop_quality*self.event_density[event])
                        
        return np.clip(total_quality, 1e-15, 10).astype('float128')
        
    def ComputeEventDensity(self, target):
        x, y = np.mgrid[0:self.map_size[0]:self.grid_size[0], 0:self.map_size[1]:self.grid_size[1]]
        xy = np.column_stack([x.flat, y.flat])
        mu = np.array(target[0])
        sigma = np.array([target[1], target[1]])
        covariance = np.diag(sigma**2)
        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
        event = z.reshape(x.shape)
            
        #return np.ones(self.size)

        return event/np.sum(event)

    def Norm(self, arr):

        sum = 0

        for i in range(len(arr)):
            sum += arr[i]**2

        return np.sqrt(sum)
    
    class DEBUGTOOL:
        
        def __init__(self, id):
            self.id = id
        
        def PlotSensorVoronoi(self, role, voronoi_graph):
            #print(self.id, ": ", np.unique(voronoi_graph))
            plt.imshow(voronoi_graph)
            plt.title("Sensor Voronoi of " + role)
            plt.show()
        
def KillCB(msg):
    global kill
    if msg.data == 1:
        kill = True
        
def FailureCB(msg):
    global failure
    if msg.data == 1:
        failure = True
    
if __name__ == "__main__":
    global kill
    global failure
    
    kill = False
    failure = False
    
    rospy.init_node('control_manager', anonymous=False, disable_signals=True)
    
    r = rospy.get_param("/rate", "60")
    rate = rospy.Rate(float(r))
    
    
    # Environment Parameters
    grid_size   = rospy.get_param("/grid_size", 0.1)
    grid_size   = np.array([grid_size, grid_size])
    map_width   = rospy.get_param("/map_width", 24)
    map_height  = rospy.get_param("/map_height", 24)
    map_size    = np.array([map_height, map_width])

    # General Agent's Settings
    id                      = rospy.get_param("~id", default=0)
    
    np.random.seed(id + 5)
    pos_x                   = rospy.get_param("~pos_x", default=0)
    pos_y                   = rospy.get_param("~pos_y", default=0)
    init_position           = np.array([pos_x, pos_y])#(np.random.random((1,2))*3)[0]# + 9#
    max_speed               = rospy.get_param("~max_speed", default=1)
    camera_valid            = rospy.get_param("~camera", default=0)
    manipulator_valid       = rospy.get_param("~manipulator", default=0)
    smoke_detector_valid    = rospy.get_param("~smoke_detector", default=0)
    valid_sensor            = {'camera'         : camera_valid,
                                'manipulator'    : manipulator_valid,
                                'smoke_detector' : smoke_detector_valid}
    general_info = {'id':           id,
                    'position':     init_position,
                    'valid_sensor': valid_sensor,
                    'max_speed':    max_speed}
    
    if camera_valid:
        # Camera Settings
        per_x               = rospy.get_param("~per_x", default=0)
        per_y               = rospy.get_param("~per_y", default=0)
        init_perspective    = np.array([float(per_x), float(per_y)])
        angle_of_view       = rospy.get_param("~angle_of_view")
        range_limit         = rospy.get_param("~desired_range")
        camera_variance     = rospy.get_param("~camera_variance", default=1)
        
        camera_info         = { 'perspective'    : init_perspective,
                                'angle of view'  : angle_of_view,
                                'desired range'  : range_limit,
                                'variance'       : camera_variance}
    else:
        camera_info = None
    
    if manipulator_valid:
        # Manipulator Settings
        arm_length          = rospy.get_param("~arm_length", default = 1)
        approx_param        = rospy.get_param("~approx_param", default=20)
        
        manipulator_info    = { 'arm length'     : arm_length,
                                'k'              : approx_param}
    else:
        manipulator_info = None

    if smoke_detector_valid:
        # Smoke Detector Settings
        smoke_variance      = rospy.get_param("~smoke_variance", default=1)
        smoke_detector_info = { 'variance'       : smoke_variance } 
    else:
        smoke_detector_info = None

    UAV_self = PTZCamera(map_size = map_size, grid_size = grid_size, general_properties=general_info,
                        camera_properties=camera_info, smoke_detector_properties=smoke_detector_info, 
                        manipulator_properties=manipulator_info, coop = True, balance = True, strength = 10000)
    
    rospy.Subscriber("/kill", Int16, KillCB)
    rospy.Subscriber("/agent_"+str(id)+"/failure", Int16, FailureCB)
    
    frame = []
    score = []
    pos_x = []
    pos_y = []
    cnt = 0
    
    while not rospy.is_shutdown() and not kill and not failure:
        UAV_self.Update()
        
        frame.append(cnt)
        score.append(UAV_self.total_score)
        pos_x.append(UAV_self.pos[0])
        pos_y.append(UAV_self.pos[1])
        cnt += 1
        
        rate.sleep()
    
    plt_dict = {'frame_id'              : frame,
                str(id)+"'s score"      : score,
                "pos_x"                 : pos_x,
                "pos_y"                 : pos_y}
            
    df = pd.DataFrame.from_dict(plt_dict) 
    df.to_csv (r"~/research_ws/src/voronoi_cbsa/result/8/coop/balance_coop_"+str(id)+".csv", index=False, header=True)
