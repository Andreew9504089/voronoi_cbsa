#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, Point
from voronoi_cbsa.msg import ExchangeData, NeighborInfoArray, TargetInfoArray, SensorArray, Sensor
from std_msgs.msg import Int16, Float32MultiArray, Int16MultiArray, Float32, Float64, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
from math import cos, acos, sqrt, exp, sin
from time import time, sleep
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import itertools

class PTZCamera():
    def __init__(self, map_size, grid_size, general_properties, camera_properties=None, 
                 manipulator_properties=None, smoke_detector_properties=None, K_p=1, K_v=1):

        self.K_p = K_p
        self.K_v = K_v
        
        # Setting up environment parameters
        self.total_agents       = rospy.get_param('/total_agents', 1)
        self.total_sensors      = rospy.get_param('/total_sensors', 2)
        self.visualize_option   = rospy.get_param('/visualize_option', default = -1)
        self.grid_size          = grid_size
        self.map_size           = map_size
        self.size               = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
        
        # Setting up agent's general parameters
        self.id                 = general_properties['id']
        self.pos                = general_properties['position']
        self.valid_sensors      = general_properties['valid_sensor']
        self.sensor_qualities   = {}

        
        if self.valid_sensors['camera']:
            # Camera property
            self.perspective        = camera_properties['perspective']/self.Norm(camera_properties['perspective'])
            self.camera_range       = camera_properties['range_limit']
            self.angle_of_view      = camera_properties['Angel_of_view']
            self.camera_variance    = camera_properties['variance']

        if self.valid_sensors['manipulator']:
            # Manipulator property
            self.operation_range    = manipulator_properties['Arm_length']
            self.approx_param       = manipulator_properties['k']
        
        if self.valid_sensors['smoke detector']:
            # Smoke Detector property
            self.smoke_variance = smoke_detector_properties['variance']
        
        self.target_received        = False
        self.targets                = {}
        
        self.event_density          = {}
        self.event_density_buffer   = {}
        
        self.neighbors              = []
        self.role                   = {}
        self.neighbors_buffer       = {}
        self.role_buffer            = {}
        
        self.sensor_graph           = {}
        self.sensor_voronoi         = {}
        
        for sensor in self.valid_sensors.keys():
            self.sensor_graph[sensor]       = []
            self.sensor_voronoi[sensor]     = np.zeros(self.size)   
            self.sensor_qualities[sensor]   = {}   
            self.role[sensor]               = self.valid_sensors[sensor]
            
        self.RosInit()

    def RosInit(self):

        rospy.Subscriber("local/neighbor_info", NeighborInfoArray, self.NeighborCallback)
        rospy.Subscriber("local/target", TargetInfoArray, self.TargetCallback)
        rospy.Subscriber("local/role", SensorArray, self.RoleCallback)
        
        
        self.pub_pos                = rospy.Publisher("local/position", Point, queue_size=10)
        self.pub_exchange_data      = rospy.Publisher("local/exchange_data",ExchangeData, queue_size=10)
        
        self.pub_total_score        = rospy.Publisher("visualize/total_score", Float64, queue_size=10)
        self.pub_sensor_scores      = rospy.Publisher("visualize/sensor_scores", Float64MultiArray, queue=10)
        self.pub_pose               = rospy.Publisher("visualize/pose", Pose, queue_size=10)
        self.pub_sub_voronoi        = rospy.Publisher("visualize/voronoi", Int16MultiArray, queue_size=10)
    
    def NeighborCallback(self, msg):
        self.neighbors_buffer = {}
        
        for neighbor in msg.neighbors:

            pos_x   = neighbor.position.x
            pos_y   = neighbor.position.y
            pos     = np.array([pos_x, pos_y])
            role    = {}
            
            for sensor in neighbor.role:
                role[sensor.type] = sensor.score

            self.neighbors_buffer[neighbor.id] = {"position":   pos, "role": role}
    
    def TargetCallback(self, msg):
        self.target_received        = True
        self.target_buffer          = {}
        self.event_density_buffer   = {}

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

            self.target_buffer[target.id]           = [pos, std, weight, vel, target.id, requirements]
            self.event_density_buffer[target.id]    = self.ComputeEventDensity(target = self.target_buffer[target.id])

        for target in self.event_density.keys():
            if target not in self.target_buffer.keys():
                self.target_buffer[target] = np.zeros(self.size)
            
    def RoleCallback(self, msg):
        
        for sensor in self.valid_sensors.keys():
            if self.valid_sensors[sensor]:
                self.role_buffer[sensor] = 0
              
        for sensor in msg.sensors:
            self.role_buffer[sensor.type] = sensor.score

    def PublishInfo(self):
        pos     = Point()
        pos.x   = self.pos[0]
        pos.y   = self.pos[1]
        self.pub_pos.publish(pos)

        total_score  = Float64()

        total_score.data = self.total_score
        self.pub_total_score.publish(total_score)
        
        sensor_scores = Float64MultiArray
        sensor_scores.data = [self.sensor_scores[role] for role in self.valid_sensors.keys()]
        
        data = ExchangeData()

        data.id = self.id
        data.position.x = self.pos[0]
        data.position.y = self.pos[1]
        
        sensor_arr = SensorArray()
        sensor_arr.data = []
        for i in self.valid_sensors.keys():
            sensor = Sensor()
            sensor.type = i
            sensor.score = self.valid_sensors[i]
            sensor_arr.data.append(sensor)
            
        data.role = sensor_arr

        self.pub_exchange.publish(data)
        
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

        if self.target_received:
            self.neighbors          = self.neighbors_buffer.copy()                      
            self.targets            = self.target_buffer.copy()   
            self.event_density      = self.event_density_buffer.copy()                        
            self.role               = self.role_buffer
            
            self.sensor_qualities.clear()
            
            for role in list(self.valid_sensors.keys).extend([-1]):
                self.sensor_voronoi[role] = np.zeros(self.size)
                self.sensor_graph[role] = self.ComputeNeighbors(role=role)
                self.UpdateSensorVoronoi(role = role)

            u_p, u_v = self.ComputeControlSignal()
            self.UpdatePosition(u_p)
            self.UpdatePerspective(u_v)
            self.PublishInfo()
            
    def UpdatePosition(self, u_p):
        
        self.pos += self.K_p * u_p
        
        if self.pos[0] < 0:
            self.pos[0] = 0
        if self.pos[1] < 0:
            self.pos[1] = 0
            
        return
    
    def UpdatePerspective(self, u_v):
        
        self.perspective += self.K_v*u_v
        self.perspective /= self.Norm(self.perspective)
    
    def UpdateSensorVoronoi(self, role):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')

        global_event = np.zeros(self.size)
        for target in self.targets:
            if role in target[5]:
                global_event += self.event_density[target]
        
        if self.role[role] > 0:
            pos_self = self.pos
            grid_size = self.grid_size
            
            total_cost = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*global_event
            sensor_voronoi = np.full(self.size, self.id)
            
        else:
            total_cost = np.full(self.size, np.inf)
            sensor_voronoi = np.full(self.size, -1)
            
        for neighbor in self.sensor_graph[role]:

            pos_self = self.neighbors[neighbor]["position"]
            grid_size = self.grid_size

            cost = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*global_event#self.ComputeCost(role, pos_self, grid_size, x_coords, y_coords)
            sensor_voronoi = np.where(cost < total_cost, neighbor, sensor_voronoi)
            total_cost[sensor_voronoi] = 0
            
        self.sensor_voronoi[role] = sensor_voronoi
    
    ## 2023/6/26 Checkpoint 
    def ComputeUtility(self):
        self.total_score    = 0
        self.sensor_scores  = {}
        
        for event in self.targets.keys():
            for role in self.valid_sensors.keys():
                tmp = 0
                if self.valid_sensors[role] and role in event[5]:
                    tmp = 1 + self.ComputeSelfQuality(role=role)
                    
                    for k in self.valid_sensors.keys():
                        if k in event[5] and k != role:
                            tmp *= 1 + self.ComputeCooperateQuality(coop_role = role, role=k, event = event, eval = True)
                    
                    tmp *= self.event_density[event]
                    
                    self.total_score += np.sum(tmp)
                self.sensor_scores[role] += np.sum(tmp)
                    
    def ComputeControlSignal(self):
        u_x = 0
        u_y = 0
        for event in self.targets.keys():
            for role in self.valid_sensors.keys():
                if self.valid_sensors[role] and role in event[5]:
                    tmp_x = self.ComputeGradient(role, 'x')
                    tmp_y = self.ComputeGradient(role, 'y')
                    
                    for k in self.valid_sensors.keys():
                        if k in event[5] and k != role:
                            coop_quality = self.ComputeCooperateQuality(coop_role = role, role=k)
                            tmp_x *= 1 + coop_quality
                            tmp_y *= 1 + coop_quality
                            
                tmp_x_1 = tmp_x
                tmp_y_1 = tmp_y
                
                for i in self.valid_sensors.keys():
                    if self.valid_sensors[i] and i != role:
                        tmp_x = self.ComputeGradient(i, 'x')*(1 + self.ComputeSelfQuality(role=role))
                        tmp_y = self.ComputeGradient(i, 'y')*(1 + self.ComputeSelfQuality(role=role))
                        
                        for k in self.valid_sensors.keys():
                            if k in event[5] and k != i and k != role:
                                coop_quality = self.ComputeCooperateQuality(coop_role = role, role=k)
                                tmp_x *= 1 + coop_quality
                                tmp_y *= 1 + coop_quality
                
                tmp_x_1 += tmp_x
                tmp_y_1 += tmp_y
                                
                tmp_x_1 *= self.event_density[event]
                tmp_y_1 *= self.event_density[event]
        
                u_x += np.sum(tmp_x_1)
                u_y += np.sum(tmp_y_1)
                    
        u_v = np.array([0,0])
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        pos_self = self.pos
        grid_size = self.grid_size
        vec = np.dstack((x_coords, y_coords))/grid_size - pos_self

        for event in self.targets.keys():
            if self.valid_sensors['camera'] and 'camera' in event[5]:
                dist    = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
                tmp_v   = vec/(dist*(1-cos(self.angle_of_view)))
            
            u_v += np.sum(tmp_v)
            
        return np.array([u_x, u_y]), u_v
              
    def ComputeGradient(self, role, type):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        pos_self = self.pos
        grid_size = self.grid_size
        
        if role == "camera":
            dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)-self.camera_range
            gradient = (-dist*exp(-(dist**2)/(2*(self.camera_variance**2)))/(self.camera_variance**2))
            gradient = np.where(self.sensor_voronoi[role] == self.id, 0, gradient)
            
        elif role == "manipulator":
            dist = self.operation_range - np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
            gradient = -(2*self.approx_param*exp(-2*self.approx_param*dist))/((1+exp(-2*self.approx_param*dist))**2)
            gradient = np.where(self.sensor_voronoi[role] == self.id, 0, gradient)
            
        elif role == "smoke detector":
            dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
            gradient = -dist*exp(-(dist**2)/(2*(self.smoke_variance**2)))/(self.smoke_variance**2)
            gradient = np.where(self.sensor_voronoi[role] == self.id, 0, gradient)
            
        gradient = gradient * 2 * (pos_self[0] - x_coords) if type == 'x' else gradient * 2 * (pos_self[1] - y_coords) 
        
        return gradient
    
    def ComputeSelfQuality(self, role):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        pos_self = self.pos
        grid_size = self.grid_size

        self_territory = np.where(self.sensor_voronoi[role] == self.id, -1, self.sensor_voronoi[role])
        
        if role == 'manipulator':
            dist = self.operation_range - np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
            self_quality = 1/(1+exp(-2*self.approx_param*(role)))
            self_quality = np.where(self_territory > -1, 0, self_quality)
        
        elif role == 'smoke detector':
            dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
            self_quality = exp(-(dist**2)/(2*self.smoke_variance**2))
            self_quality = np.where(self_territory > -1, 0, self_quality)
            
        elif role == 'camera':
            dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)-self.camera_range
            self_quality = exp(-(dist**2)/(2*self.camera_variance**2))
            self_quality = np.where(self_territory > -1, 0, self_quality)
        
        return self_quality
    
    def ComputeCooperateQuality(self, coop_role, role, event, eval = False):
        x_coords, y_coords = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        pos_self = self.pos
        grid_size = self.grid_size
        # vec = np.dstack((x_coords, y_coords))/grid_size - pos_self

        if role == 'camera':
            
            total_quality = np.zeros(self.size)
            for neighbor in list(self.sensor_graph[role].keys())+[self.id]:
                pos = self.neighbors[neighbor]["position"] if neighbor != self.id else self.pos
                dist = np.sqrt((pos[0]/grid_size[0] - x_coords)**2 + (pos[1]/grid_size[1] - y_coords)**2)-self.camera_range

                territory = np.where(self.sensor_voronoi[role] == neighbor, -1, self.sensor_voronoi[role])
                individual_quality = exp(-(dist**2)/(2*self.camera_variance**2))
                individual_quality = np.where(territory > -1, 0, individual_quality)
                
                total_quality += individual_quality
                
                if eval:               
                    self_territory = np.where(self.sensor_voronoi[coop_role] == self.id, -1, self.sensor_voronoi[coop_role])

                    if coop_role == 'manipulator':
                        coop_dist = self.operation_range - np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
                        self_quality = 1/(1+exp(-2*self.approx_param*(coop_dist)))
                        self_quality = np.where(self_territory > -1, 0, self_quality)
                        self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                    
                    elif coop_role == 'smoke detector':
                        coop_dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
                        self_quality = exp(-(dist**2)/(2*self.smoke_variance**2))
                        self_quality = np.where(self_territory > -1, 0, self_quality)                    
                        self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                
        elif role == 'manipulator':
            total_quality = np.zeros(self.size)
            for neighbor in list(self.sensor_graph[role].keys())+[self.id]:
                pos = self.neighbors[neighbor]["position"] if neighbor != self.id else self.pos
                dist = self.operation_range - np.sqrt((pos[0]/grid_size[0] - x_coords)**2 + (pos[1]/grid_size[1] - y_coords)**2)

                territory = np.where(self.sensor_voronoi[role] == neighbor, -1, self.sensor_voronoi[role])
                individual_quality = 1/(1+exp(-2*self.approx_param*(dist)))
                individual_quality = np.where(territory > -1, 0, individual_quality)
                
                total_quality += individual_quality
                
                if eval:           
                    self_territory = np.where(self.sensor_voronoi[coop_role] == self.id, -1, self.sensor_voronoi[coop_role])

                    if coop_role == 'camera':
                        coop_dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)-self.camera_range
                        self_quality = exp(-(dist**2)/(2*self.camera_variance**2))
                        self_quality = np.where(self_territory > -1, 0, self_quality)
                        self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])

                    elif coop_role == 'smoke detector':
                        coop_dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
                        self_quality = exp(-(dist**2)/(2*self.smoke_variance**2))
                        self_quality = np.where(self_territory > -1, 0, self_quality)
                        self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                
                   
        elif role == 'smoke detector':
            total_quality = np.zeros(self.size)
            for neighbor in list(self.sensor_graph[role].keys())+[self.id]:
                pos = self.neighbors[neighbor]["position"] if neighbor != self.id else self.pos
                dist = np.sqrt((pos[0]/grid_size[0] - x_coords)**2 + (pos[1]/grid_size[1] - y_coords)**2)

                territory = np.where(self.sensor_voronoi[role] == neighbor, -1, self.sensor_voronoi[role])
                individual_quality = exp(-(dist**2)/(2*self.smoke_variance**2))
                individual_quality = np.where(territory > -1, 0, individual_quality)
                
                total_quality += individual_quality
                
                if eval:          
                    self_territory = np.where(self.sensor_voronoi[coop_role] == self.id, -1, self.sensor_voronoi[coop_role])

                    if coop_role == 'manipulator':
                        coop_dist = self.operation_range - np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)
                        self_quality = 1/(1+exp(-2*self.approx_param*(coop_dist)))
                        self_quality = np.where(self_territory > -1, 0, self_quality)
                        self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                    
                    elif coop_role == 'camera':
                        coop_dist = np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2)-self.camera_range
                        self_quality = exp(-(dist**2)/(2*self.camera_variance**2))
                        self_quality = np.where(self_territory > -1, 0, self_quality)
                        self.sensor_qualities[coop_role][neighbor][event] += np.sum(individual_quality*self_quality*self.event_density[event])
                
        return total_quality
        
    def ComputeEventDensity(self, target):
        x, y = np.mgrid[0:self.map_size[0]:self.grid_size[0], 0:self.map_size[1]:self.grid_size[1]]
        xy = np.column_stack([x.flat, y.flat])
        mu = np.array(target[0])
        sigma = np.array([target[1], target[1]])
        covariance = np.diag(sigma**2)
        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
        event = z.reshape(x.shape)
            
        return event

    def Norm(self, arr):

        sum = 0

        for i in range(len(arr)):
            sum += arr[i]**2

        return sqrt(sum)

if __name__ == "__main__":
    rospy.init_node('control_manager', anonymous=True, disable_signals=True)
    
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
    pos_x                   = rospy.get_param("~pos_x", default=0)
    pos_y                   = rospy.get_param("~pos_y", default=0)
    init_position           = np.array([pos_x, pos_y])#(np.random.random((1,2))*18)[0]
    camera_valid            = rospy.get_param("~camera", default=0)
    manipulator_valid       = rospy.get_param("~manipulator", default=0)
    smoke_detector_valid    = rospy.get_param("~smoke_detector", default=0)
    valid_sensor            = {'camera'         : camera_valid,
                                'manipulator'    : manipulator_valid,
                                'smoke detector' : smoke_detector_valid}
    general_info = {'id': id,
                    'position': init_position,
                    'valid_sensor': valid_sensor}
    
    if camera_valid:
        # Camera Settings
        per_x               = rospy.get_param("~per_x", default=0)
        per_y               = rospy.get_param("~per_y", default=0)
        init_perspective    = np.array([per_x, per_y])
        angle_of_view       = rospy.get_param("~angle_of_view")
        range_limit         = rospy.get_param("~desired_range")
        camera_variance     = rospy.get_param("~camera_variance", default=1)
        
        camera_info         = { 'perspective'    : init_perspective,
                                'angle of view'  : angle_of_view,
                                'desired range'  : range_limit,
                                'variance'       : camera_variance}
    else:
        caemra_info = None
    
    if manipulator_valid:
        # Manipulator Settings
        arm_length          = rospy.get_param("~arm_length", default = 1)
        approx_param        = rospy.get_param("~approx_param", default=20)
        
        manipulator_info    = { 'arm_length'     : arm_length,
                                'k'              : approx_param}
    else:
        manipulator_info = None

    if smoke_detector_valid:
        # Smoke Detector Settings
        smoke_variance      = rospy.get_param("~smoke_variance", default=1)
        smoke_detector_info = { 'variance'       : smoke_variance } 
    else:
        smoke_detector_info = None

    UAV_self = PTZCamera(camera_info, map_size = map_size, grid_size = grid_size, general_properties=general_info,
                            camera_properties=camera_info, smoke_detector_properties=smoke_detector_info)
    
    while not rospy.is_shutdown():
        UAV_self.Update()
        #print(UAV_self.pos)
        rate.sleep()