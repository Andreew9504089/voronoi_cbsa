#! /usr/bin/env python
import rospy
import pygame
import numpy as np
import matplotlib.pyplot as plt

import time
from std_msgs.msg import Int16MultiArray, Float32MultiArray
from geometry_msgs.msg import PointStamped

class Visualize2D():
    def __init__(self):
        self.total_agents = rospy.get_param('/total_agents', '1')
        self.agent_neighbors = {}
        self.agent_control_mode = {}
        self.agent_scores = {}
        self.agent_pos = {}
        self.agent_step = {}
        self.color = {}
        self.start = 0
        self.tmp = [-1 for i in range(self.total_agents)]
        self.switch = [0 for i in range(self.total_agents)]
        self.controller = [1 for i in range(self.total_agents)]
        self.t = 0
        self.tmp_score = 0
        self.plt_utl = []
        self.plt_time = []

        color_pool = [(255, 0, 0), (255, 128, 0), (255,255,0), (0,255,0), (0,255,255), (0,0,255), (178,102,255), (255,0,255)]
        for i in range(self.total_agents):
            try:
                self.color[i] = color_pool[i]
            except:
                self.color[i] = list(np.random.choice(range(255),size=3))
            rospy.Subscriber("/agent"+str(i)+"/visualize/neighbors", Int16MultiArray, self.NeighborCB)
            rospy.Subscriber("/agent"+str(i)+"/visualize/control_mode", Int16MultiArray, self.ControlModeCB)
            rospy.Subscriber("/agent"+str(i)+"/visualize/scores", Float32MultiArray, self.ScoresCB)
            rospy.Subscriber("/agent"+str(i)+"/visualize/position", PointStamped, self.PositionCB)
            rospy.Subscriber("/agent"+str(i)+"/visualize/step", Int16MultiArray, self.StepCB)
        
        self.grid_size = np.array((0.1, 0.1))
        self.size = np.array([20,20])/self.grid_size
        self.window_size = self.size*4
        self.display = pygame.display.set_mode(self.window_size)
        self.display.fill((0,0,0))
        self.blockSize = int(self.window_size[0]/self.size[0])

    def NeighborCB(self, msg):
        agent_id = msg.data[0]
        neighbors = []

        for i in range(1, len(msg.data)):
            neighbors.append(msg.data[i])

        self.agent_neighbors[agent_id] = neighbors

    def ControlModeCB(self, msg):
        agent_id = msg.data[0]

        self.agent_control_mode[agent_id] = msg.data[1]

    def ScoresCB(self, msg):
        agent_id = int(msg.data[0])

        self.agent_scores[agent_id] = msg.data[1]

    def PositionCB(self, msg):
        agent_id = int(msg.header.frame_id)

        self.agent_pos[agent_id] = np.asarray([msg.point.x, msg.point.y])

    def StepCB(self, msg):
        agent_id = int(msg.data[0])

        self.agent_step[agent_id] = msg.data[1]

    def Update(self):

        total_utility = [0.,0.]
    
        self.display.fill((0,0,0))

        for id in range(self.total_agents):
            if self.start == 0 and len(self.agent_pos.keys()) > 0:
                self.start = time.time()

            if id in self.agent_pos.keys() and id in self.agent_control_mode.keys():

                if self.agent_control_mode[id] != self.tmp[id]:
                    self.t = np.round(time.time() - self.start,2)
                    self.tmp[id] = self.agent_control_mode[id]
                
                if self.controller[id] != self.agent_control_mode[id]:
                    self.controller[id] = self.agent_control_mode[id]
                    self.switch[id] += 1

                width = 0 if self.agent_control_mode[id] == 1 else 2
                pygame.draw.circle(self.display, self.color[id], 
                                   self.agent_pos[id]/self.grid_size*self.blockSize, radius=10, width=width)
            
                if id in self.agent_scores.keys():
                    score = self.agent_scores[id]
                    total_utility[0] += self.agent_scores[id] if self.agent_control_mode[id] == 1 else 0
                    total_utility[1] += self.agent_scores[id] if self.agent_control_mode[id] == -1 else 0
                    center = self.agent_pos[id]/self.grid_size*self.blockSize
                    font = pygame.font.Font('freesansbold.ttf', 15)
                    text = font.render(str(id)+": "+str(np.round(score, 3)), True, self.color[id])
                    textRect = text.get_rect()
                    textRect.center = (center[0], center[1] - 20)
                    self.display.blit(text, textRect)

                if id in self.agent_step.keys():
                    step = "     "+str(self.agent_step[id])
                    center = self.agent_pos[id]/self.grid_size*self.blockSize
                    font = pygame.font.Font('freesansbold.ttf', 15)
                    text = font.render(step, True, self.color[id])
                    textRect = text.get_rect()
                    textRect.center = (center[0], center[1] - 40)
                    self.display.blit(text, textRect)


                if id in self.agent_neighbors.keys():
                    pos = self.agent_pos[id]/self.grid_size*self.blockSize
                    for neighbor in self.agent_neighbors[id]:
                        if neighbor in self.agent_pos.keys():
                            neighbor_pos = self.agent_pos[neighbor]/self.grid_size*self.blockSize
                            pygame.draw.line(self.display, (255,255,255), pos, neighbor_pos, 2)
        
        for id in range(self.total_agents):
            if id in self.agent_neighbors.keys() and id in self.agent_pos.keys():
                color = self.color[id]
                pos = self.agent_pos[id]/self.grid_size*self.blockSize
                for neighbor in self.agent_neighbors[id]:
                    if neighbor in self.agent_pos.keys():
                        neighbor_pos = self.agent_pos[neighbor]/self.grid_size*self.blockSize
                        pygame.draw.line(self.display, color, pos, (pos+neighbor_pos)/2, 2)
               
        font = pygame.font.Font('freesansbold.ttf', 25)
        text = font.render(str(np.round(total_utility[0]*total_utility[1], 3)),True, (100,175,255))
        textRect = text.get_rect()
        textRect.center = (80, 20)
        self.display.blit(text, textRect)

        if np.round(total_utility[0]*total_utility[1], 3) != self.tmp_score:
            self.tmp_score = np.round(total_utility[0]*total_utility[1], 3)
            print("score:"+str(np.round(total_utility[0]*total_utility[1], 3)))
            self.plt_time.append(np.round(time.time()-self.start, 2))
            self.plt_utl.append(np.round(total_utility[0]*total_utility[1], 3))
        
        font = pygame.font.Font('freesansbold.ttf', 20)
        text = font.render(str(self.t),True, (100,175,255))
        textRect = text.get_rect()
        textRect.center = (80, 40)
        self.display.blit(text, textRect)

        font = pygame.font.Font('freesansbold.ttf', 20)
        text = font.render(str(np.sum(self.switch)),True, (100,175,255))
        textRect = text.get_rect()
        textRect.center = (80, 60)
        self.display.blit(text, textRect)

        pygame.display.flip()

if __name__=="__main__":
    rospy.init_node('visualizer', anonymous=False, disable_signals=True)
    total_agents = rospy.get_param('/total_agents', '1')
    pygame.init()

    visualizer = Visualize2D()
    Done = False
    while not Done:
        for op in pygame.event.get():
            if op.type == pygame.QUIT:
                Done = True 
        
        visualizer.Update()
            
    pygame.quit()

    plt.plot(visualizer.plt_time, visualizer.plt_utl)
    plt.title("NonConsensus "+str(total_agents)+" agents")
    plt.xlabel('Time')
    plt.ylabel('Utility')
    plt.show()