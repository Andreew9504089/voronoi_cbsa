#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray, Int16MultiArray, Float32MultiArray, Int16, Float32
from geometry_msgs.msg import PointStamped, Point
from voronoi_cbsa.msg import ExchangeDataArray, VoteList, ExchangeData, SensorArray, Sensor

import random
import numpy as np
import time
import sys
from scipy .spatial import Delaunay
import matplotlib.pyplot as plt

class RoleManager():
    def __init__(self, id, grid_size = (0.1, 0.1), total_agents = 3, 
                    pos = (np.random.random((1,2))*20)[0],
                    valid_sensors = [0,1]):
        
        self.grid_size = grid_size
        self.total_agents = total_agents
        self.total_sensors = rospy.get_param('/total_sensors', 2)
        self.communication_range = np.inf
        self.neighbors = []
        self.valid_members = {}                                 # {id: (pos, age, bid, vote_list)}
        self.valid_members_buffer = {}                          # Buffer to store information that is received during each CBSA iteration
        self.pos = pos
        self.role = 0
        self.id = id
        self.reward = 0
        self.bid = 0
        self.bid_list = {self.id:   self.bid}
        self.ratio = 0
        self.weight = 0
        self.valid_sensors = valid_sensors
        self.switched_role = self.role
        
        self.pos_init = False
        self.pos_buffer = pos
        self.scores_init = False
        self.scores = {}
        self.scores_buffer = {}
        
        for sensor in self.valid_sensors:
            self.scores_buffer[sensor] = 0
            self.scores[sensor] = 0
        
        self.scores2pub = self.scores
        
        if self.id == 0 or self.id == 1 or self.id == 2 :
            self.role = 0
        else:
            self.role = 1
            
        self.ROSInit()
    
    def ROSInit(self):
        rospy.Subscriber("local/received_exchange_data", ExchangeDataArray, self.ExchangeCallback)
        rospy.Subscriber("local/position", Point, self.PositionCallback)
        rospy.Subscriber("local/scores", Float32MultiArray, self.ScoresCallback)
        rospy.Subscriber("local/weight", Float32, self.WeightCallback)

        self.pub_neighbors      = rospy.Publisher("visualize/neighbors", Int16MultiArray, queue_size=10)
        self.pub_control_mode   = rospy.Publisher("visualize/control_mode", Int16MultiArray, queue_size=10)
        self.pub_scores         = rospy.Publisher("visualize/scores", Float32MultiArray, queue_size=10)
        self.pub_position       = rospy.Publisher("visualize/position", PointStamped, queue_size=10)
        self.pub_step           = rospy.Publisher("visualize/step", Int16MultiArray, queue_size=10)
        self.pub_role           = rospy.Publisher("local/role", SensorArray, queue_size=10)
        self.pub_exchange       = rospy.Publisher("local/exchange_data", ExchangeData, queue_size=10)

    def WeightCallback(self, msg):
        self.weight = msg.data
        
    def PositionCallback(self, msg):
        self.pos_init = True
        self.pos_buffer = np.array([msg.x, msg.y])

    def ScoresCallback(self, msg):
        self.scores_init = True

        for i, score in enumerate(msg.data):
            self.scores_buffer[i] = score
            self.scores2pub[i] = score
            
    def ExchangeCallback(self, msg):
        # Retrieve the other agents' information who are within the communication range
        self.valid_members_buffer = {}
        for data in msg.data:

            info = {}
            valid_id = data.id
                    
            info["pos"]         = np.array([data.position.x, data.position.y])
            info["bid"]         = data.bid
            info["score"]       = data.score
            info["role"]        = data.role
            info["weight"]      = data.weight

            self.valid_members_buffer[valid_id] = info
    
    def ComputeNeighbors(self):

        if len(self.valid_members.keys()) >= 4:

            keys = [self.id]
            for key in self.valid_members.keys():
                keys.append(key)

            idx_map = {}
            idx_list = []

            for i, key in enumerate(keys):
                idx_map[key] = i
                idx_list.append(key)

            points = [0 for i in keys]
            points[0] = self.pos/self.grid_size

            for member in self.valid_members.keys():
                pos = np.asarray([self.valid_members[member]["pos"][0], self.valid_members[member]["pos"][1]])
                points[idx_map[member]] = pos/self.grid_size

            points = np.asarray(points)
            tri = Delaunay(points)

            ids = []
            for simplex in tri.simplices:
                if idx_map[self.id] in simplex:
                    for id in simplex:
                        
                        ids.append(id)

            self.neighbors = []
            for member in self.valid_members.keys():
                if idx_map[member] in ids:
                    if np.linalg.norm(self.valid_members[member]["pos"] - self.pos) <= (self.weight + self.valid_members[member]["weight"]):
                        self.neighbors.append(member)
            
        elif len(self.valid_members.keys()) >= 0:

            self.neighbors = []
            for member in self.valid_members.keys():
                if np.linalg.norm(self.valid_members[member]["pos"] - self.pos) <= (self.weight + self.valid_members[member]["weight"]):
                    self.neighbors.append(member)

        else:
            self.neighbors = []           
    
    def ComputeReward(self):
        if len(self.neighbors) > 0:
            
            capacities = {}
            potential_capacity = {}
            for sensor in self.valid_sensors:
                potential_capacity[sensor] = 0
                capacities[sensor] = 0
                      
            for sensor in self.valid_sensors:
                cnt = 0
                for neighbor in self.neighbors:
                    
                    neighbor_pos = np.asarray([self.valid_members[neighbor]["pos"][0], self.valid_members[neighbor]["pos"][1]])
                    distance = np.linalg.norm(self.pos - neighbor_pos)
                    
                    potential_capacity[sensor] += np.exp(-0.5*(distance**2))*self.valid_members[neighbor]["score"]\
                                                    if self.valid_members[neighbor]["role"] == self.role else 0
                    
                    cnt += 1 if self.valid_members[neighbor]["role"] == self.role else 0  
                cnt = 1                              
                potential_capacity[sensor] = 1/(1 + potential_capacity[sensor]/cnt)
                capacities[sensor] = self.scores[sensor]/potential_capacity[sensor]
            
            best_role = max(capacities, key=capacities.get)
            reward = capacities[best_role] - capacities[self.role]
            
            if reward > 0:
                self.switched_role = best_role
                return reward
            else:
                self.switched_role = self.role
                return 0
            
        else:

            best_role = max(self.scores, key=self.scores.get)
            reward = self.scores[best_role] - self.scores[self.role]
            
            if reward > 0:
                self.switched_role = best_role
                return reward
            else:
                self.switched_role = self.role
                return 0
            
    def PublishInfo(self, step=-1):

        def PublishExchangeData():
            data = ExchangeData()

            data.id = self.id
            data.bid = float(self.bid)
            data.position.x = self.pos[0]
            data.position.y = self.pos[1]
            data.score = self.scores[self.role]
                
            data.role = self.role
            data.weight = self.weight

            self.pub_exchange.publish(data)

        def PublishPosition():
            position = PointStamped()
            position.header.frame_id = str(self.id)
            position.point.x = self.pos[0]
            position.point.y = self.pos[1]

            self.pub_position.publish(position)

        def PublishControlMode():
            control_mode = Int16MultiArray()
            control_mode.data = [self.id, self.role]

            self.pub_control_mode.publish(control_mode)
        def PublishNeighbors():
            neighbors = Int16MultiArray()
            neighbors.data = [self.id]
            
            for i in range(len(self.neighbors)):
                neighbors.data.append(self.neighbors[i])
            
            self.pub_neighbors.publish(neighbors)
            
        def PublishScores():
            scores = Float32MultiArray()
            scores.data = [self.id, self.scores2pub[self.role]]

            self.pub_scores.publish(scores)
            
        def PublishStep():

            data = Int16MultiArray()
            data.data = [self.id, 0]

            self.pub_step.publish(data)
            
        def PublishRole():
            roles = SensorArray()
            data = []
            for i in range(self.total_sensors):
                s = Sensor()
                if self.role == i:
                    score = 1.
                else:
                    score = 0.
                s.type = i
                s.score = score
                data.append(s)
            roles.sensors = data
            self.pub_role.publish(roles)

        PublishStep()
        PublishExchangeData()
        PublishControlMode()
        PublishNeighbors()
        PublishPosition()
        PublishScores()
        PublishRole()
        
    def Run(self, type = "CBSA"):
        
        r = rospy.get_param("/rate", "60")
        rate = rospy.Rate(float(r))
        # Initialize
        self.PublishInfo()
        self.valid_members = self.valid_members_buffer.copy()
        self.scores = self.scores_buffer.copy()
        self.pos = self.pos_buffer
        cnt = 0
        step = -1
        
        while not rospy.is_shutdown():

            if self.pos_init and self.scores_init:

                # Reset agent's knowledge and bid
                self.ComputeNeighbors()
                self.reward = self.ComputeReward()
                self.bid = self.reward
                self.valid_members.update(self.valid_members_buffer.copy())
                self.scores.update(self.scores_buffer.copy())
                self.pos = self.pos_buffer
                self.PublishInfo(step)

                while True:

                    if type == "Ego":

                        best_role = max(self.scores, key=self.scores.get)
                        if self.role != best_role:
                            self.role = best_role
                            step = cnt

                        self.PublishInfo(step)
                        self.valid_members.update(self.valid_members_buffer.copy())
                        self.scores.update(self.scores_buffer.copy())
                        self.pos = self.pos_buffer
                        break
                    
                    elif type == "NonConsensus":

                        if self.reward > 0:
                            self.role = self.switched_role 
                            step = cnt
                        
                        self.PublishInfo(step)
                        self.valid_members.update(self.valid_members_buffer.copy())
                        self.scores.update(self.scores_buffer.copy())
                        self.pos = self.pos_buffer

                        break
                    
                    elif type == "Consensus":
                        # Contrsuct bid list
                        self.bid_list = {}
                        self.bid_list[self.id] = self.bid
                        for id in self.neighbors:
                            self.bid_list[id] = self.valid_members[id]["bid"] 

                        # Use self.vote_list to perform voting
                        self.vote = max(self.bid_list, key=self.bid_list.get)
                        
                        if self.vote == self.id and self.reward > 0:
                            #self.role = self.switched_role
                            step = cnt

                        self.PublishInfo(step)
                        self.valid_members.update(self.valid_members_buffer.copy())
                        self.scores.update(self.scores_buffer.copy())
                        self.pos = self.pos_buffer
                        
                        break
                            
            cnt += 1
            #time.sleep(2)
            rate.sleep()

if __name__=="__main__":
    rospy.init_node('role_manager', anonymous=True, disable_signals=True)
    
    id      = rospy.get_param("~id", default=0)
    pos_x   = rospy.get_param("~x", default=0)
    pos_y   = rospy.get_param("~y", default=0)
    
    pos     = np.array([pos_x, pos_y])
    
    total_agents = int(rospy.get_param('/total_agents', '1'))

    #agent = RoleManager(id = int(id), total_agents= total_agents)
    #agent.Run(type = "Consensus")
    
    #camera = rospy.get_param()    
        
            
                                                    