#! /usr/bin/env python3
import rospy
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import imageio

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray, Float32MultiArray, Float64MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from voronoi_cbsa.msg import ExchangeData, NeighborInfoArray, TargetInfoArray
from scipy.stats import multivariate_normal
import itertools

class Visualize2D():
    def __init__(self):
        self.total_agents = rospy.get_param('/total_agents', 1)
        map_width   = rospy.get_param("/map_width", 24)
        map_height  = rospy.get_param("/map_height", 24)
        self.map_size    = np.array([map_height, map_width])
        grid_size = rospy.get_param("/grid_size", 0.1)
        self.grid_size = np.array([grid_size, grid_size])
        self.size = (self.map_size/self.grid_size).astype(np.int64)
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
        self.FoV = {}
        self.FoV_init = False
        self.sub_voronoi_init = False
        self.global_voronoi_init = False
        self.target_received = False 

        color_pool = [(255, 0, 0), (255, 128, 0), (255,255,0), (0,255,0), (0,255,255), (0,0,255), (178,102,255), (255,0,255)]
        for i in range(self.total_agents):
            try:
                self.color[i] = color_pool[i]
            except:
                self.color[i] = list(np.random.choice(range(255),size=3))
            rospy.Subscriber("/agent_"+str(i)+"/visualize/neighbors", Int16MultiArray, self.NeighborCB)
            rospy.Subscriber("/agent_"+str(i)+"/visualize/control_mode", Int16MultiArray, self.ControlModeCB)
            rospy.Subscriber("/agent_"+str(i)+"/visualize/scores", Float32MultiArray, self.ScoresCB)
            rospy.Subscriber("/agent_"+str(i)+"/visualize/position", PointStamped, self.PositionCB)
            rospy.Subscriber("/agent_"+str(i)+"/visualize/step", Int16MultiArray, self.StepCB)
            rospy.Subscriber("/agent_"+str(i)+"/FoV", Image, self.FoVCB(id=i))
            rospy.Subscriber("/target", TargetInfoArray, self.TargetCallback)
            # rospy.Subscriber("/agent_"+str(i)+"/visualize/sub_voronoi", Int16MultiArray, self.SubVoronoiCB)
            # rospy.Subscriber("/agent_"+str(i)+"/visualize/global_voronoi", Int16MultiArray, self.GlobalVoronoiCB)
            
        self.window_size = self.size*4
        self.display = pygame.display.set_mode(self.window_size)
        self.display.fill((0,0,0))
        self.blockSize = int(self.window_size[0]/self.size[0])

        # define the codec and create a video writer object
        self.cnt = 0
        self.images = []
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.prefix = '~/research_ws/src/voronoi_cbsa/result/'
        if not os.path.exists(self.prefix + self.timestr):
            os.makedirs(self.prefix + self.timestr)

    def TargetCallback(self, msg):
        self.target_received = True
        self.target_buffer = []

        for target in msg.targets:
            pos_x = target.position.x
            pos_y = target.position.y
            pos = np.array([pos_x, pos_y])

            std = target.standard_deviation
            weight = target.weight

            vel_x = target.velocity.linear.x
            vel_y = target.velocity.linear.y
            vel = np.array([vel_x, vel_y])

            self.target_buffer.append([pos, std, weight, vel, target.id])
            
    def FoVCB(self, id):
        def callback(msg):
            self.FoV_init = True
            
            bridge = CvBridge()
            img_array = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.FoV[id] = (img_array/255).astype(np.float32)
            
        return callback

    def SubVoronoiCB(self, msg):
        self.sub_voronoi_init = True
        self.sub_voronoi = np.reshape(np.asarray(msg.data), self.size)
    
    def GlobalVoronoiCB(self, msg):
        self.global_voronoi_init = True
        self.global_voronoi = np.reshape(np.asarray(msg.data), self.size)
    
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

        def PlotFoV(event):
            for i in range(self.total_agents):
                if  i in self.agent_control_mode:
                    if i in self.FoV.keys():
                        if i == 0:
                            FoV_plot = np.expand_dims(self.FoV[i], axis = 0)
                        else:
                            FoV_plot = np.concatenate([FoV_plot, np.expand_dims(self.FoV[i], axis = 0)])
                    
                    else:
                        if i == 0:
                            FoV_plot = np.expand_dims(np.full((event.shape), 0), axis = 0)
                        else:
                            FoV_plot = np.concatenate([FoV_plot, np.expand_dims(np.full((event.shape), 0), axis = 0)])
                else:
                    if i == 0:
                            FoV_plot = np.expand_dims(np.full((event.shape), 0), axis = 0)
                    else:
                        FoV_plot = np.concatenate([FoV_plot, np.expand_dims(np.full((event.shape), 0), axis = 0)])
                        
            plt = np.argmax(FoV_plot, axis = 0)
            tmp = np.max(FoV_plot, axis = 0)
            mask = np.where(tmp == 0.0)
            plt[mask] = -1
            
            return plt
        
        self.display.fill((0,0,0))
        
        if self.target_received:
            event = ComputeEventDensity(self.map_size, self.target_buffer, self.grid_size)
            event_plt = ((event - event.min()) * (1/(event.max() - event.min()) * 255)).astype('uint8')
            
            FoV_plt = PlotFoV(event_plt)
            for x_map,x in enumerate(range(0, self.window_size[0], self.blockSize)):
                for y_map,y in enumerate(range(0, self.window_size[1], self.blockSize)):
                    rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                    pygame.draw.rect(self.display, (event_plt[x_map,y_map],event_plt[x_map,y_map],event_plt[x_map,y_map]), rect, 1)
                    if FoV_plt[y_map, x_map] in self.FoV.keys():
                        id = FoV_plt[y_map, x_map]
                        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                        w = 0.3
                        color = [0, 0, 0]
                        color[0] = w*self.color[id][0] + (1-w)*event_plt[x_map,y_map]
                        color[1] = w*self.color[id][1] + (1-w)*event_plt[x_map,y_map]
                        color[2] = w*self.color[id][2] + (1-w)*event_plt[x_map,y_map]
                        pygame.draw.rect(self.display, color, rect, 1)
                    
                    else:
                        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                        pygame.draw.rect(self.display, (event_plt[x_map,y_map],event_plt[x_map,y_map],event_plt[x_map,y_map]), rect, 1)
            
        total_utility = [0.,0.]
            
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
            #print("score:"+str(np.round(total_utility[0]*total_utility[1], 3)))
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
        
        # transform the pixels to the format used by open-cv
        pixels = cv2.rotate(pygame.surfarray.pixels3d(self.display), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        self.images.append(pixels)
        
    def End(self):
        imageio.mimsave(self.prefix+self.timestr+'.gif', self.images)
        
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
                visualizer.End()
        
        visualizer.Update()
            
    pygame.quit()

    plt.plot(visualizer.plt_time, visualizer.plt_utl)
    plt.title("NonConsensus "+str(total_agents)+" agents")
    plt.xlabel('Time')
    plt.ylabel('Utility')
    plt.show()