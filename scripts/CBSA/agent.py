#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int16MultiArray, Float32MultiArray, Int16
from geometry_msgs.msg import PointStamped, Point
from voronoi_cbsa.msg import ExchangeDataArray, VoteList, ExchangeData

import random
import numpy as np
import time
import sys
from scipy .spatial import Delaunay
import matplotlib.pyplot as plt

global test

class RoleManager():
    def __init__(self, id, grid_size = (0.1, 0.1), max_age = 1000, total_agents = 3, 
                    pos = (np.random.random((1,2))*20)[0],
                    scores = {1:    random.random()*100, -1:   random.random()*100}, test = False):
        self.grid_size = grid_size
        self.total_agents = total_agents
        self.communication_range = np.inf

        self.lock = False                                       # Only update information after each vote-consensus iter
        self.neighbors = []
        self.valid_members = {}                                 # {id: (pos, age, bid, vote_list)}
        self.valid_members_buffer = {}                          # Buffer to store information that is received during each CBSA iteration
        self.max_age = max_age                                  # maximum steps that others' information will last
        self.pos = pos
        self.role = 1                             
        self.id = id                                            # Unique ID for each agent
        self.stable = True                                      # Determine whether the vote list is stable, start switching if true
        self.score = scores
        self.reward = self.ComputeReward()
        self.bid = self.reward                                  # self.bid = utility(switched state) - utility(current state)
        self.bid_list = {self.id:   self.bid}                   # bid list stores voronoi neighbors' bid
        self.vote = -1                                          # indicator for which agent to vote for
        self.vote_list = {self.id:  self.vote}                  # vote list records agent's own votes
        self.merge_list = {}                                    # merge list merges all agents' votes
        self.neighbor_vote_lists = []                           # stores neighbors' vote lists
        
        if not test:
            self.pos_init = False
            self.pos_buffer = pos
            self.scores_init = False
            self.scores_buffer = {-1: 0, 1: 0}
            self.score2pub = {-1: 0, 1: 0}

        
        else:
            self.pos_init = True
            self.scores_init = True
            self.pos_buffer = pos
            self.scores_buffer = scores
            self.pub_position_test = rospy.Publisher("local/position", Point, queue_size=10)

        self.ROSInit()
        # print(self.id, " => ", self.score)

    # Initialize ROS node, suscriber and publisher
    def ROSInit(self):
        rospy.Subscriber("local/received_exchange_data", ExchangeDataArray, self.ExchangeCallback)
        rospy.Subscriber("local/position", Point, self.PositionCallback)
        rospy.Subscriber("local/scores", Float32MultiArray, self.ScoresCallback)

        self.pub_neighbors      = rospy.Publisher("visualize/neighbors", Int16MultiArray, queue_size=10)
        self.pub_control_mode   = rospy.Publisher("visualize/control_mode", Int16MultiArray, queue_size=10)
        self.pub_scores         = rospy.Publisher("visualize/scores", Float32MultiArray, queue_size=10)
        self.pub_position       = rospy.Publisher("visualize/position", PointStamped, queue_size=10)
        self.pub_step           = rospy.Publisher("visualize/step", Int16MultiArray, queue_size=10)
        self.pub_role           = rospy.Publisher("local/role", Int16, queue_size=10)
        self.pub_exchange       = rospy.Publisher("local/exchange_data", ExchangeData, queue_size=10)

    def PositionCallback(self, msg):
        self.pos_init = True
        self.pos_buffer = np.array([msg.x, msg.y])

    def ScoresCallback(self, msg):
        self.scores_init = True
        for i, score in enumerate(msg.data):
            self.scores_buffer[(-1)**i] = score if score > 1 else 0
            self.score2pub[(-1)**i] = score
        
    # Receive neighbors' id, bid and vote list
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
            
            tmp = {}
            for i in range(len(data.vote_list)):
                idx = data.vote_list[i].index
                vote = data.vote_list[i].vote
                tmp[idx] = vote

            info["vote_list"]   = tmp

            self.valid_members_buffer[valid_id] = info
                
    # Find the agent's voronoi neighbors
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

                    self.neighbors.append(member)
            
        elif len(self.valid_members.keys()) >= 0:

            self.neighbors = []
            for member in self.valid_members.keys():
                self.neighbors.append(member)

        else:
            self.neighbors = []
        
        self.neighbor_vote_lists = []
        for id in self.valid_members.keys():

            # Store the neighbors' data for further purpose
            if id in self.neighbors:
                self.neighbor_vote_lists.append(self.valid_members[id]["vote_list"]) 

    def ComputeReward(self, type="Centralized"):
        if type == "Centralized":
            if len(self.neighbors) > 0:

                neighbor_scores = [0, 0]
                tmp = []
                for neighbor in self.neighbors:
                    neighbor_pos = np.asarray([self.valid_members[neighbor]["pos"][0], self.valid_members[neighbor]["pos"][1]])
                    distance = np.linalg.norm(self.pos - neighbor_pos)
                    tmp.append(distance)
                    
                    neighbor_scores[0] += np.exp(-2*distance)*self.valid_members[neighbor]["score"] \
                                            if self.valid_members[neighbor]["role"] == 1 else 0
                    neighbor_scores[1] += np.exp(-2*distance)*self.valid_members[neighbor]["score"] \
                                            if self.valid_members[neighbor]["role"] == -1 else 0
                distance = np.mean(tmp)
                current_utility = (1 + np.exp(-2*distance)*self.score[self.role] + neighbor_scores[0])*neighbor_scores[1] \
                                        if self.role == 1 \
                                            else (1 + np.exp(-2*distance)*self.score[self.role] + neighbor_scores[1])*neighbor_scores[0] 
                
                switched_utility = (1 + np.exp(-2*distance)*self.score[self.role*-1] + neighbor_scores[0])*neighbor_scores[1] \
                                        if self.role == -1 \
                                            else (1 + np.exp(-2*distance)*self.score[self.role*-1] + neighbor_scores[1])*neighbor_scores[0]
                
                return switched_utility - current_utility
            
            else:

                return self.score[self.role*-1] - self.score[self.role]
        
        elif type == "Consensus":
            pass
        
    # Perform Voting Phase with neighbors' bids and vote for the highest one
    def VotingPhase(self):
        
        # Contrsuct bid list
        self.bid_list = {}
        self.bid_list[self.id] = self.bid
        for id in self.neighbors:
            self.bid_list[id] = self.valid_members[id]["bid"] 

        # Use self.vote_list to perform voting
        self.vote = max(self.bid_list, key=self.bid_list.get)
        self.vote_list = {}

        # Construct agent's vote list with the one being voted as True and False otherwise
        iter = self.neighbors + [self.id]
        for id in iter:
            if id == self.vote and self.bid_list[id] > 0:          
                self.vote_list[id] = True
            else:
                self.vote_list[id] = False
        
        self.merge_list.update(self.vote_list)


    # Fuse information based on neighbors' vote list
    def ConsensusPhase(self):

        if len(self.neighbor_vote_lists) > 0:
            self.stable = True
        else:
            self.stable = False

        # Process the recieved neighbors' vote list to perform consensus fusion
        for d in self.neighbor_vote_lists:
            for id, value in d.items():
                if id in self.vote_list:
                    
                    # Check if the agent agrees on all other agents' votes
                    if (self.vote_list[id] != value):
                        self.stable = False
                        self.merge_list[id] = self.vote_list[id] | value

                elif id not in self.merge_list:
                    self.stable = False
                    self.merge_list[id] = value
        
        # Check if agent and its neighbors have conflicts or not
        for id in self.neighbors:
            if self.merge_list[id] & self.merge_list[self.id]:
                self.ResolveConflict(id)   

    # Resolve conflicts when the neighbor and itself are both voted
    def ResolveConflict(self, id):
        if id in self.neighbors:
            if self.bid < self.bid_list[id]:
                self.bid = 0
        else:
            return

    def PublishInfo(self, step=-1):

        def PublishExchangeData():
            data = ExchangeData()

            data.id = self.id
            data.bid = self.bid
            data.position.x = self.pos[0]
            data.position.y = self.pos[1]
            data.score = self.score[self.role]
            data.role = self.role

            vote_list = []
            for i,key in enumerate(self.vote_list.keys()):
                tmp = VoteList()
                tmp.index = key
                tmp.vote = bool(self.vote_list[key])
                data.vote_list.append(tmp)
                vote_list.append(tmp)

            self.pub_exchange.publish(data)

        def PublishPosition():
            position = PointStamped()
            position.header.frame_id = str(self.id)
            position.point.x = self.pos[0]
            position.point.y = self.pos[1]

            self.pub_position.publish(position)

            if test:
                position = Point()
                position.x = self.pos[0]
                position.y = self.pos[1]
                self.pub_position_test.publish(position)

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
            scores.data = [self.id, self.score2pub[self.role]]

            self.pub_scores.publish(scores)
        def PublishStep():

            data = Int16MultiArray()
            data.data = [self.id, 0]

            self.pub_step.publish(data)

        PublishStep()
        PublishExchangeData()
        PublishControlMode()
        PublishNeighbors()
        PublishPosition()
        PublishScores()

        self.pub_role.publish(self.role)

    def Run(self, type = "CBSA"):
        global test
        
        r = rospy.get_param("/rate", "60")
        rate = rospy.Rate(float(r))
        # Initialize
        self.PublishInfo()
        self.valid_members = self.valid_members_buffer.copy()
        self.score = self.scores_buffer.copy()
        self.pos = self.pos_buffer if test == True else self.pos
        cnt = 0
        step = -1
        
        while not rospy.is_shutdown():

            if self.pos_init and self.scores_init:

                # Reset agent's knowledge and bid
                self.ComputeNeighbors()
                self.reward = self.ComputeReward()
                self.bid = self.reward
                self.valid_members.update(self.valid_members_buffer.copy())
                self.score.update(self.scores_buffer.copy())
                self.pos = self.pos_buffer
                self.PublishInfo(step)

                # print(self.id, " => ", self.role, " => ", np.round(self.bid,4), " => ", self.neighbors)

                while True:
                
                    if type == "2hop-Consensus":
                        # Perform CBSA
                        self.ComputeNeighbors()
                        self.VotingPhase()
                        self.ConsensusPhase()
                        
                        self.PublishInfo(step)
                        self.valid_members.update(self.valid_members_buffer.copy())
                        self.score.update(self.scores_buffer.copy())
                        self.pos = self.pos_buffer if test == True else self.pos
                        
                        # If the agent agrees on others, then switch base on the merge_list and reset CBSA round
                        if self.stable:
                            # If the agreement (merge list) indicates that the agent should switch, then switch
                            if self.merge_list[self.id] and self.reward > 0: 
                                self.role *= -1 
                                step = cnt

                            self.merge_list = {}
                            self.vote_list = {}
                            
                            break

                    elif type == "Ego":

                        if self.score[self.role*-1] > self.score[self.role]:
                            self.role *= -1 
                            step = cnt

                        self.PublishInfo(step)
                        self.valid_members.update(self.valid_members_buffer.copy())
                        self.score.update(self.scores_buffer.copy())
                        self.pos = self.pos_buffer if test == True else self.pos

                        break
                    
                    elif type == "NonConsensus":

                        if self.reward > 0:
                            self.role *= -1 
                            step = cnt
                        
                        self.PublishInfo(step)
                        self.valid_members.update(self.valid_members_buffer.copy())
                        self.score.update(self.scores_buffer.copy())
                        self.pos = self.pos_buffer if test == True else self.pos

                        break
                    
                    elif type == "non-voting":
                        # Contrsuct bid list
                        self.bid_list = {}
                        self.bid_list[self.id] = self.bid
                        for id in self.neighbors:
                            self.bid_list[id] = self.valid_members[id]["bid"] 

                        # Use self.vote_list to perform voting
                        self.vote = max(self.bid_list, key=self.bid_list.get)
                        
                        if self.vote == self.id and self.reward > 0:
                            self.role *= -1
                            step = cnt
                        
                        # if self.score[1] == 0:
                        #     self.role = -1
                            
                        self.PublishInfo(step)
                        self.valid_members.update(self.valid_members_buffer.copy())
                        self.score.update(self.scores_buffer.copy())
                        self.pos = self.pos_buffer if test == True else self.pos
                        
                        break
                            

            cnt += 1
            #time.sleep(2)
            rate.sleep()

if __name__=="__main__":
    rospy.init_node('role_manager', anonymous=True, disable_signals=True)
    
    test = rospy.get_param("/test", default=0)
    test = False if test == 0 else True
    id      = rospy.get_param("~id", default=0)
    pos_x   = rospy.get_param("~x", default=0)
    pos_y   = rospy.get_param("~y", default=0)
    
    pos     = np.array([pos_x, pos_y])
    
    total_agents = int(rospy.get_param('/total_agents', '1'))

    if test:
        scores = {0: {1: 78.5,  -1: 81.4},
                1: {1: 85.0,  -1: 91.2},
                2: {1: 85.0,  -1: 91.1},
                3: {1: 90.3,  -1: 93.0},
                4: {1: 44.3,  -1: 96.8},
                5: {1: 70.1,  -1: 81.58},
                6: {1: 19.56, -1: 25.78},
                7: {1: 56.57, -1: 69.68},
                8: {1: 31.39, -1: 44.04},
                9: {1: 45.93, -1: 50.17},
                10: {1: 35.84, -1: 87.4},
                11: {1: 16.3, -1: 72.45},
                12: {1: 76.54, -1: 95.7},
                13: {1: 43.56, -1: 57.46},
                14: {1: 27.54, -1: 83.56},
                15: {1: 25.94, -1: 32.69},
                16: {1: 19.54, -1: 21.47},
                17: {1: 15.44, -1: 42.18},
                18: {1: 62.08, -1: 71.27},
                19: {1: 51.59, -1: 90.40},}
        
        if int(id) in scores:
            agent = RoleManager(id = int(id), total_agents= total_agents, pos = pos, scores = scores[int(id)], test=test)
        
        else:
            agent = RoleManager(id = int(id), total_agents= total_agents, pos = pos, test=test)
    
    else:
        agent = RoleManager(id = int(id), total_agents= total_agents,test=test)
   
    agent.Run(type = "non-voting")