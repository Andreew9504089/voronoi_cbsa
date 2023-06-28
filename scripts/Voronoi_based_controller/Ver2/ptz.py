#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, Point
from voronoi_cbsa.msg import ExchangeData, NeighborInfoArray, TargetInfoArray
from std_msgs.msg import Int16, Float32MultiArray, Int16MultiArray, Float32
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
    def __init__(self, properties, map_size, grid_size,
                    Kv = 30, Ka = 5, Kp = 2, step = 0.1):

        self.total_agents = rospy.get_param('/total_agents', '1')
        self.grid_size = grid_size
        self.map_size = map_size
        self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
        self.id = properties['id']
        self.pos = properties['position']
        self.perspective = properties['perspective']/self.Norm(properties['perspective'])
        self.alpha = properties['AngleofView']/180*np.pi
        self.R = properties['range_limit']
        self.lamb = properties['lambda']
        self.color = properties['color']
        self.max_speed = properties['max_speed']
        self.perspective_force = 0
        self.zoom_force = 0
        self.positional_force = np.array([0.,0.])
        self.targets = []
        self.FoV = np.zeros(self.size)
        self.intercept_map = np.zeros(self.size)
        self.global_event = np.zeros(self.size)
        self.global_event_plt = np.zeros(self.size)
        self.global_voronoi = np.zeros(self.size)
        self.local_Voronoi = []
        self.Kv = Kv
        self.Ka = Ka
        self.Kp = Kp
        self.step = step
        self.neighbors = []
        self.map_plt = np.zeros((self.size))
        self.intercept_quality = 0
        self.coverage_quality = 0
        self.last_pos = self.pos
        self.role = 1                           # 1: Intercepetor -1: observer
        self.neighbors_buffer = {}
        self.role_buffer = "Tracker"
        self.target_received = False
        self.neighbor_FoV = {}
        self.neighbor_intercept_map = {}

        self.RosInit()

    def RosInit(self):

        rospy.Subscriber("local/neighbor_info", NeighborInfoArray, self.NeighborCallback)
        rospy.Subscriber("local/target", TargetInfoArray, self.TargetCallback)
        rospy.Subscriber("local/role", Int16, self.RoleCallback)
        
        for i in range(self.total_agents):
            rospy.Subscriber("/agent_"+str(i)+"/FoV", Image, self.FoVCB(id=i))
            rospy.Subscriber("/agent_"+str(i)+"/InterceptMap", Image, self.InterceptCB(id=i))
            
        self.pub_scores             = rospy.Publisher("local/scores", Float32MultiArray, queue_size=10)
        self.pub_pos                = rospy.Publisher("local/position", Point, queue_size=10)
        self.pub_fov                = rospy.Publisher("FoV", Image, queue_size=10)
        self.pub_int_map            = rospy.Publisher("InterceptMap", Image, queue_size=10)
        self.pub_sub_voronoi        = rospy.Publisher("visualize/sub_voronoi", Int16MultiArray, queue_size=10)
        self.pub_global_voronoi     = rospy.Publisher("visualize/global_voronoi", Int16MultiArray, queue_size=10)
        self.pub_weight             = rospy.Publisher("local/weight", Float32, queue_size=10)  

    def FoVCB(self, id):
        def callback(msg):
            self.FoV_init = True
            
            bridge = CvBridge()
            img_array = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.neighbor_FoV[id] = (img_array/255).astype(np.float32)
            
        return callback
    
    def InterceptCB(self, id):
        def callback(msg):
            self.FoV_init = True
            
            bridge = CvBridge()
            img_array = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.neighbor_intercept_map[id] = (img_array/255).astype(np.float32)
            
        return callback
    
    def NeighborCallback(self, msg):
        self.neighbors_buffer = {}

        for neighbor in msg.neighbors:
            pos_x = neighbor.position.x
            pos_y = neighbor.position.y
            pos = np.array([pos_x, pos_y])

            role = "Tracker" if neighbor.role == -1 else "Interceptor"

            self.neighbors_buffer[neighbor.id] = {"position":   pos, "role": role}
    
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

    def RoleCallback(self, msg):
        self.role_buffer = "Tracker" if msg.data == -1 else "Interceptor"

    def PublishInfo(self):
        pos = Point()
        pos.x = self.pos[0]
        pos.y = self.pos[1]
        self.pub_pos.publish(pos)

        scores = Float32MultiArray()

        scores.data = [self.intercept_quality, self.coverage_quality]
        self.pub_scores.publish(scores)
        
        bridge = CvBridge()
        if self.role == "Tracker":
            FoV = (((self.FoV)*255).astype(np.uint8))
        else:
            FoV = ((self.intercept_map)*255).astype(np.uint8)
        msg = bridge.cv2_to_imgmsg(FoV, encoding='mono8')
        msg.header.stamp = rospy.Time.now()
        self.pub_fov.publish(msg)
        
        weight = Float32()
        if self.role == "Tracker":
            weight.data = self.R*np.cos(self.alpha)
        else:
            weight.data = self.max_speed
        self.pub_weight.publish(weight)
        
    # Find the agent's voronoi neighbors
    def ComputeNeighbors(self):

        if len(self.neighbors.keys()) >= 4:

            keys = [self.id]
            for key in self.neighbors.keys():
                keys.append(key)

            idx_map = {}
            idx_list = []

            for i, key in enumerate(keys):
                idx_map[key] = i
                idx_list.append(key)

            points = [0 for i in keys]
            points[0] = self.pos/self.grid_size

            for member in self.neighbors.keys():
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
            for member in self.neighbors.keys():
                if idx_map[member] in ids:

                    neighbors.append(member)
            
        elif len(self.neighbors.keys()) >= 0:

            neighbors = []
            for member in self.neighbors.keys():
                neighbors.append(member)

        else:
            neighbors = []
            
        return neighbors
    
    def Update(self, targets=None, neighbors=None, centroid_tmp=None, geo_center_tmp=None, sub_global_voronoi=None):

        if self.target_received:
            self.neighbors = self.neighbors_buffer.copy()                      # neighbors
            self.targets = self.target_buffer.copy()                           # targets
            self.role = self.role_buffer

            vor_neighbors = self.ComputeNeighbors()
            tmp = {}
            for id in vor_neighbors:
                tmp[id] = self.neighbors[id]
            self.neighbors = tmp
            # self.centroid = centroid_tmp
            # self.geo_center = geo_center_tmp
            # self.sub_global = sub_global_voronoi

            self.global_event = self.ComputeEventDensity(targets = self.targets) 
            self.global_event_plt = ((self.global_event - self.global_event.min()) * (1/(self.global_event.max()
                                        - self.global_event.min()) * 255)).astype('uint8')
            
            if self.role == "Tracker":
                self.UpdateGlobalVoronoi()
            elif self.role == "Interceptor":
                self.UpdateSubGlobalVoronoi()

            self.FoV = np.zeros(self.size)
            self.UpdateFoV()
            self.UpdateLocalVoronoi()
            self.ComputeLocalCentroidal()
            self.UpdateOrientation()
            self.UpdateZoomLevel()

            # self.UpdateRole()
            self.UpdatePosition()
            self.PublishInfo()
        
        #   if self.role == "Interceptor":
        #     print(self.id, ": ", self.intercept_quality, ", ", self.coverage_quality)

    def UpdateOrientation(self):

        self.perspective += self.perspective_force*self.step
        self.perspective /= self.Norm(self.perspective)

        return

    def UpdateZoomLevel(self):

        self.alpha += self.zoom_force*self.step

        return

    def UpdatePosition(self):
        
        # Tracker Control Law (Move to Sweetspot)

        if self.role == "Tracker":
            
            centroid_force = (self.centroid*self.grid_size - self.pos) * (1 - self.R*cos(self.alpha)
                                /np.linalg.norm((self.centroid*self.grid_size - self.pos)))
            
            rot = np.array([[cos(np.pi/2), -sin(np.pi/2)],
                        [sin(np.pi/2), cos(np.pi/2)]])
            v = rot@self.perspective.reshape(2,1)

            allign_force = np.reshape(((self.geo_center*self.grid_size - self.pos)@v)*v, (1,2))[0]

            self.positional_force = np.clip((centroid_force+allign_force)/2, -self.max_speed, self.max_speed)

        # Interceptor Control Law (Move to Centroid)

        elif self.role == "Interceptor":

            self.positional_force = np.clip((self.centroid*self.grid_size - self.pos), -self.max_speed, self.max_speed)

        self.pos += self.Kp*self.positional_force*self.step
        if self.pos[0] < 0:
            self.pos[0] = 0
        if self.pos[1] < 0:
            self.pos[1] = 0

        return

    def UpdateGlobalVoronoi(self):

        x_coords, y_coords = np.meshgrid(np.arange(self.global_event.shape[0]), np.arange(self.global_event.shape[1]), indexing='ij')

        pos_self = self.pos
        grid_size = self.grid_size
        global_voronoi = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*self.global_event
        sub_global_voronoi = global_voronoi.copy()

        for neighbor in self.neighbors.keys():

            if self.neighbors[neighbor]["role"] == "Tracker":
                pos_self = self.neighbors[neighbor]["position"]
                grid_size = self.grid_size

                cost = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*self.global_event
                mask = np.where(cost < global_voronoi)
                global_voronoi[mask] = 0

            pos_self = self.neighbors[neighbor]["position"]
            grid_size = self.grid_size

            cost = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*self.global_event
            mask = np.where(cost < sub_global_voronoi)
            sub_global_voronoi[mask] = 0

        indices = np.where(global_voronoi > 0)
        if np.sum(self.global_event[indices]) > 0:
            weighted_x = int(np.average(indices[0], weights=self.global_event[indices]))
            weighted_y = int(np.average(indices[1], weights=self.global_event[indices]))
            
            self.geo_center = ((int(np.mean(indices[0])), int(np.mean(indices[1]))))
        else:
            weighted_x = 0
            weighted_y = 0
            
            self.geo_center = (0.,0.)
            
        self.centroid = ((weighted_x, weighted_y))
        self.sub_voronoi = np.where(sub_global_voronoi > 0, 1, 0)
        self.global_voronoi = global_voronoi

        return 
    
    def UpdateSubGlobalVoronoi(self):
        x_coords, y_coords = np.meshgrid(np.arange(self.global_event.shape[0]), np.arange(self.global_event.shape[1]), indexing='ij')

        pos_self = self.pos
        grid_size = self.grid_size
        global_voronoi = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*self.global_event
        sub_global_voronoi = global_voronoi.copy()

        for neighbor in self.neighbors.keys():

            if self.neighbors[neighbor]["role"] == "Interceptor":
                pos_self = self.neighbors[neighbor]["position"]
                grid_size = self.grid_size

                cost = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*self.global_event
                mask = np.where(cost < global_voronoi)
                global_voronoi[mask] = 0

            pos_self = self.neighbors[neighbor]["position"]
            grid_size = self.grid_size

            cost = (np.sqrt((pos_self[0]/grid_size[0] - x_coords)**2 + (pos_self[1]/grid_size[1] - y_coords)**2))*self.global_event
            mask = np.where(cost < sub_global_voronoi)
            sub_global_voronoi[mask] = 0

        indices = np.where(global_voronoi > 0)
        if np.sum(self.global_event[indices]) > 0:
            weighted_x = int(np.average(indices[0], weights=self.global_event[indices]))
            weighted_y = int(np.average(indices[1], weights=self.global_event[indices]))
            
            self.geo_center = ((int(np.mean(indices[0])), int(np.mean(indices[1]))))
        else:
            weighted_x = 0
            weighted_y = 0
            
            self.geo_center = (0.,0.)
            
        self.centroid = ((weighted_x, weighted_y))
        self.sub_voronoi = np.where(sub_global_voronoi > 0, 1, 0)
        self.global_voronoi = global_voronoi

    def UpdateFoV(self):

        range_max = (self.lamb + 1)/(self.lamb)*self.R
        quality_map = None
        quality_int_map = None
        self.intercept_map = np.zeros(self.FoV.shape)

        for y_map in range(max(int((self.pos[1] - range_max)/self.grid_size[1]), 0),\
                            min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[1])):

            x_map = np.arange(max(int((self.pos[0] - range_max)/self.grid_size[0]), 0),\
                            min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0]))
            
            q_per = self.ComputePerspectiveQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
            q_res = self.ComputeResolutionQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
            q_int = self.ComputeInterceptionQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])

            quality = np.where((q_per > 0) & (q_res > 0), q_per*q_res, 0)
            q_int = np.where(q_int >= 0, q_int, 0)
            
            if quality_map is None:
                quality_map = quality
                quality_int_map = q_int
            
            else:
                quality_map = np.vstack((quality_map, quality))
                quality_int_map = np.vstack((quality_int_map, q_int))

        self.intercept_map[max(int((self.pos[1] - range_max)/self.grid_size[1]), 0):\
                                min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[0]),\
                                    max(int((self.pos[0] - range_max)/self.grid_size[0]), 0):\
                                        min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0])]\
                                            = quality_int_map
        
        self.FoV[max(int((self.pos[1] - range_max)/self.grid_size[1]), 0):\
                    min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[0]),\
                        max(int((self.pos[0] - range_max)/self.grid_size[0]), 0):\
                            min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0])]\
                                = quality_map
        return 

    def UpdateLocalVoronoi(self):
        
        quality_map = self.FoV
        intercept_map = self.intercept_map
        for neighbor in self.neighbors.keys():
            if neighbor in self.neighbor_FoV.keys():
                quality_map = np.where((quality_map > self.neighbor_FoV[neighbor]), quality_map, 0)

            if neighbor in self.neighbor_intercept_map.keys():
                intercept_map = np.where((intercept_map > self.neighbor_intercept_map[neighbor]), intercept_map, 0)
        
        #self.intercept_map = np.where(np.transpose(self.sub_voronoi) > 0, 1, 0)    
        self.coverage_quality = np.sum(quality_map*np.transpose(self.global_event))
        self.intercept_quality = np.sum(intercept_map*np.transpose(self.global_event))
        self.local_voronoi = np.array(np.where((quality_map > 0) & (self.FoV != 0))) #np.where(self.FoV != 0) 
        self.local_voronoi_map = np.where(((quality_map != 0) & (self.FoV != 0)), quality_map, 0)
        self.overlap = np.array(np.where((quality_map == 0) & (self.FoV != 0)))
        self.map_plt = np.array(np.where(quality_map != 0, self.id + 1, 0))

        return

    def ComputeLocalCentroidal(self):

        rotational_force = np.array([0.,0.]).reshape(2,1)
        zoom_force = 0

        if len(self.local_voronoi[0]) > 0:
            mu_V = 0
            v_V_t = np.array([0, 0], dtype=np.float64)
            delta_V_t = 0

            # Control law for maximizing local resolution and perspective quality
            for i in range(len(self.local_voronoi[0])):
                x_map = self.local_voronoi[1][i]
                y_map = self.local_voronoi[0][i]

                x, y = x_map*self.grid_size[0], y_map*self.grid_size[1]
                x_p = np.array([x,y]) - self.pos
                norm = self.Norm(x_p)

                if norm == 0: continue

                mu_V += ((norm**self.lamb)*self.global_event[x_map,y_map] )/(self.R**self.lamb)
                v_V_t += ((x_p)/norm)*(cos(self.alpha) - \
                                ( ( self.lamb*norm )/((self.lamb+1)*self.R)))*\
                                    ( (norm**self.lamb)/(self.R**self.lamb) )*self.global_event[x_map,y_map]
                dist = (1 - (self.lamb*norm)/((self.lamb+1)*self.R))
                dist = dist if dist >= 0 else 0
                delta_V_t += (1 - (((x_p)@self.perspective.T))/norm)\
                                *dist*((norm**self.lamb)/(self.R**self.lamb))\
                                    *self.global_event[x_map,y_map]
            
            v_V = v_V_t/mu_V
            delta_V = delta_V_t/mu_V
            delta_V = delta_V if delta_V > 0 else 1e-10
            alpha_v = acos(1-sqrt(delta_V))
            alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi
            
            rotational_force += self.Kv*(np.eye(2) - np.dot(self.perspective[:,None],\
                                            self.perspective[None,:]))  @  (v_V.reshape(2,1))
            zoom_force -= self.Ka*(self.alpha - alpha_v)

        #self.perspective_force = np.asarray([rotational_force[0][0], rotational_force[1][0]])
        self.perspective_force = ((self.centroid*self.grid_size - self.pos)\
                                    /np.linalg.norm(self.centroid*self.grid_size - self.pos)\
                                        - self.perspective)
        self.zoom_force = zoom_force

        return

    def ComputePerspectiveQuality(self, x, y):

        x_p = np.array([x,y], dtype=object) - self.pos

        return (np.matmul(x_p,self.perspective.transpose())/np.linalg.norm(x_p)
            - np.cos(self.alpha))/(1 - np.cos(self.alpha))

    def ComputeResolutionQuality(self, x, y):

        x_p = np.array([x, y], dtype=object) - self.pos

        return (((np.linalg.norm(x_p)**self.lamb)*(self.R*np.cos(self.alpha)
            - self.lamb*( np.linalg.norm(x_p) - self.R*np.cos(self.alpha)) ))
                / (self.R**(self.lamb+1)))    

    def ComputeInterceptionQuality(self, x, y):
        x_p = np.linalg.norm(np.array([x, y], dtype=object) - self.pos)
        quality = np.exp(-x_p**2/(2*(self.max_speed/2)**2))
        #quality[x_p > self.max_speed] = 0

        return quality

    def ComputeEventDensity(self, targets):

        event = []

        for i in range(len(targets)):

            x, y = np.mgrid[0:self.map_size[0]:self.grid_size[0], 0:self.map_size[1]:self.grid_size[1]]
            xy = np.column_stack([x.flat, y.flat])
            mu = np.array(targets[i][0])
            sigma = np.array([targets[i][1], targets[i][1]])
            covariance = np.diag(sigma**2)
            z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
            event.append(z.reshape(x.shape))
        
        combinations = [list(itertools.combinations(range(len(targets)), i)) for i in range(1, len(targets)+1)]
        union = np.zeros(event[0].shape)

        for c in combinations:
        
            for pair in c:

                inter = np.ones(event[0].shape)

                for i in pair:

                    inter = np.multiply(inter, event[i][:,:])

                union += ((-1)**(len(pair) + 1))*inter
            
        return union

    def Norm(self, arr):

        sum = 0

        for i in range(len(arr)):
            sum += arr[i]**2

        return sqrt(sum)


if __name__ == "__main__":
    rospy.init_node('control_manager', anonymous=True, disable_signals=True)
    
    test = rospy.get_param("/test", default=0)
    test = False if test == 0 else True
    
    if not test:
        r = rospy.get_param("/rate", "60")
        rate = rospy.Rate(float(r))
        
        grid_size = rospy.get_param("/grid_size", 0.1)
        grid_size = np.array([grid_size, grid_size])

        map_width   = rospy.get_param("/map_width", 24)
        map_height  = rospy.get_param("/map_height", 24)
        map_size    = np.array([map_height, map_width])

        id                  = rospy.get_param("~id", default=0)
        pos_x               = rospy.get_param("~pos_x", default=0)
        pos_y               = rospy.get_param("~pos_y", default=0)
        init_position       = np.array([pos_x, pos_y])#(np.random.random((1,2))*18)[0]
        per_x               = rospy.get_param("~per_x", default=0)
        per_y               = rospy.get_param("~per_y", default=0)
        init_perspective    = np.array([per_x, per_y])
        angle_of_view       = rospy.get_param("~angle_of_view")
        range_limit         = rospy.get_param("~desired_range")
        lamb                = rospy.get_param("~lambda")
        R                   = rospy.get_param("~R", default=0)
        G                   = rospy.get_param("~G", default=0)
        B                   = rospy.get_param("~B", default=0)
        color               = (R,G,B)
        max_speed           = rospy.get_param("~max_speed")

        camera_info = { 'id'            :  id,
                        'position'      :  init_position,
                        'perspective'   :  init_perspective,
                        'AngleofView'   :  angle_of_view,
                        'range_limit'   :  range_limit,
                        'lambda'        :  lamb,
                        'color'         :  color,
                        'max_speed'     :  max_speed}

        UAV_self = PTZCamera(camera_info, map_size = map_size, grid_size = grid_size)
        
        while not rospy.is_shutdown():
            UAV_self.Update()
            #print(UAV_self.pos)
            rate.sleep()