import pygame
import numpy as np
from math import cos, acos, sqrt, exp, sin
from time import sleep, time
from scipy import ndimage, sparse
from cvxopt import matrix, solvers
import osqp
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Voronoi2D():
    def __init__(self, map_size, grid_size, camera_num, \
        camera_prop, event, event_plt, target):
        self.event_plt = event_plt
        self.map = self.Map(map_size, grid_size)
        self.camera_num = camera_num
        self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
        self.PTZs = []

        for i in range(0, camera_num):
            ptz = self.PTZcamera(camera_prop[i], map_size, grid_size, target)
            self.PTZs.append(ptz)

        self.event = event

    def Update(self, step = 0.3, event_plt = None, target = []): 

        if len(target) > 0 :
            self.event_plt = event_plt
            for i in range(self.camera_num):
                self.PTZs[i].targets = target
            
        for ptz in self.PTZs:
            ptz.voronoi = []
            ptz.FoV = np.zeros(self.size)
            ptz.ComputeFoV()

        for i in range(self.camera_num):
            self.PTZs[i].UpdateVoronoi([self.PTZs[j].FoV \
                for j in range(len(self.PTZs)) if j != i])

        sum = 0
        area = 0
        cnt = [0 for i in range(len(target))]
        map_plt = np.zeros((self.map.size)) - 1
        for i in range(self.camera_num):
            map_plt = self.PTZs[i].map_plt + map_plt
            sum += self.PTZs[i].quality
            area += len(self.PTZs[i].voronoi[0])
            if self.PTZs[i].target_assigned != -1:
                cnt[self.PTZs[i].target_assigned] += 1
        print("Covered Quality: ",sum)
        # print("Total Covered Area: ", area)        

        self.map.Update(map_plt, self.PTZs, self.event_plt, target, sum)

        for i in range(self.camera_num):
            info = []
            for j in range(self.camera_num):
                if j != i:
                    info.append([self.PTZs[j].pos, self.PTZs[j].target_assigned])
            self.PTZs[i].UpdateState(info, step)

        return cnt

    def norm(self, arr):
        sum = 0
        for i in range(len(arr)):
            sum += arr[i]**2
        return sqrt(sum)
    
    class PTZcamera():
        def __init__(self, properties, map_size, grid_size, targets, \
                        Kv = 40, Kvo = 1, Ka = 5, Kao = 0, Kp = 10):
            self.grid_size = grid_size
            self.map_size = map_size
            self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
            self.id = properties['id']
            self.pos = properties['position']
            self.perspective = properties['perspective']/self.norm(properties['perspective'])
            self.alpha = properties['AngleofView']/180*np.pi
            self.R = properties['range_limit']
            self.lamb = properties['lambda']
            self.color = properties['color']
            self.translation_force = 0  # dynamics of positional changes
            self.perspective_force = 0  # dynamics of changing perspective direction
            self.alpha_dot = 0          # dynamics of zoom-in level
            self.stage = 1              # 1: Free player 2: Occupied Player 3: Cooperative player
            self.targets = targets
            self.target_assigned = -1
            self.FoV = np.zeros(self.size)
            self.Kv = Kv                # control gain for perspective control law toward voronoi cell
            self.Kvo = Kvo              # control gain for perspective control law toward overlap cell
            self.Ka = Ka                # control gain for zoom level control stems from voronoi cell
            self.Kao = Kao              # control gain for zoom level control stems from overlap cell
            self.Kp = Kp                # control gain for positional change toward voronoi cell 
            self.event = np.zeros((self.size[0], self.size[1]))

        def polygon_FOV(self):
            range_max = (self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
            R = np.array([[np.cos(self.alpha), -np.sin(self.alpha)]
                        ,[np.sin(self.alpha), np.cos(self.alpha)]])

            self.top = self.pos + range_max*self.perspective

            self.ltop = self.pos + range_max*np.reshape(R@np.reshape(self.perspective,(2,1)),(1,2))
            self.ltop = self.ltop[0]

            self.rtop = self.pos + range_max*np.reshape(np.linalg.inv(R)@np.reshape(self.perspective,(2,1)),(1,2))
            self.rtop = self.rtop[0]
        def UpdateState(self, neighbors, step):
            centroidal_forces = self.ComputeCentroidal(self.event)

            formation_force = self.FormationControl(neighbors)
            self.TargetAssignment()
            control_input = self.CBF(centroidal_forces[0], formation_force, \
                                        centroidal_forces[1], stage = self.stage)
            
            self.pos += control_input[0]*step
            self.perspective += control_input[1]*step
            self.perspective /= self.norm(self.perspective)
            self.alpha = self.alpha + centroidal_forces[2]*step

        def CBF(self, translational_force = 0, formation_force = 0, rotational_force = 0,\
                    coe = 0.8, stage = 1):

            if stage == 2:
                # td = np.asarray(self.targets[self.target_assigned][0]) - self.pos
                # td = td/self.norm(td)

                # ux = rotational_force[0]
                # uy = rotational_force[1]

                # # Define problem data
                # P = sparse.csc_matrix([[1, 0], [0, 1]])
                # q = np.array([-2*ux, -2*uy])
                # A = sparse.csc_matrix([[td[0], td[1]],[self.perspective[0], self.perspective[1]]])
                # l = np.array([cos(self.alpha) - td@self.perspective, 0])
                # u = np.array([np.inf, 0])
                # # Create an OSQP object
                # prob = osqp.OSQP()

                # # Setup workspace and change alpha parameter
                # prob.setup(P, q, A, l, u, alpha = 1, verbose=False)

                # # Solve problem
                # res = prob.solve()
                # ux = res.x[0]; uy = res.x[1]
                # cl = np.array([ux, uy])
                # n_p = (self.perspective+coe*cl)/self.norm(self.perspective+coe*cl)
                # rotational_force = cl
                res = translational_force + formation_force

                return [res, rotational_force]

            else:
                res = translational_force + formation_force
                return [res, rotational_force]

        def TargetAssignment(self):
            score = -np.inf
            self.target_assigned = -1
            pos = None

            for (target,i) in zip(self.targets,range(0,len(self.targets))):
                q_res = self.ResolutionQuality(target[0][0], target[0][1])
                q_per = self.PerspectiveQuality(target[0][0], target[0][1])
                tmp = q_res*q_per if q_res > 0 and q_per > 0 else 0
                if tmp > score and tmp > 0:
                    score = tmp
                    pos = (np.asarray(target[0])/self.grid_size).astype(np.int64)
                    self.target_assigned = i

            if pos is not None and np.any(np.all(pos == np.column_stack((self.voronoi[1],self.voronoi[0])), axis=1)):
                self.stage = 2
            else:
                self.stage = 1

            # print(self.id,"=>", self.target_assigned, "=>", score, "=>", self.stage)

        def FormationControl(self, neighbors):
            # TO-DO after Chinese New Year
            # Consider which neighbor is sharing the same target and only use them to obtain formation force
            neighbor_force = np.array([0.,0.])
            for neighbor in neighbors:
                neighbor_pos = neighbor[0]
                neighbor_target = neighbor[1]
                if neighbor_target == self.target_assigned:
                    neighbor_force += (self.pos - neighbor_pos)/(self.norm(self.pos - neighbor_pos))
            
            neighbor_norm = self.norm(neighbor_force)

            if self.stage == 2:
                target_force = (np.asarray(self.targets[self.target_assigned][0]) - self.pos)\
                                /(self.norm(np.asarray(self.targets[self.target_assigned][0])\
                                     - self.pos))
                target_norm = self.norm(target_force)

                formation_force = (target_force*(neighbor_norm/(target_norm+neighbor_norm))\
                                    + neighbor_force*(target_norm/(target_norm+neighbor_norm)))
                return formation_force

            else:
                return neighbor_force
        
        def ComputeCentroidal(self, event):
            translational_force = np.array([0.,0.])
            rotational_force = np.array([0.,0.]).reshape(2,1)
            zoom = 0
            centroid = None
            if len(self.voronoi[0]) > 0:
                mu_V = 0
                v_V_t = np.array([0, 0], dtype=np.float64)
                delta_V_t = 0
                x_center = 0
                y_center = 0

                # Control law for maximizing resolution and perspective quality
                for i in range(len(self.voronoi[0])):
                    x_map = self.voronoi[1][i]
                    y_map = self.voronoi[0][i]

                    x, y = x_map*self.grid_size[0], y_map*self.grid_size[1]
                    x_p = np.array([x,y]) - self.pos
                    norm = self.norm(x_p)
                    if norm == 0: continue
                    mu_V += ((norm**self.lamb)*event[x_map,y_map] )/(self.R**self.lamb)
                    v_V_t += ((x_p)/norm)*(cos(self.alpha) - \
                                    ( ( self.lamb*norm )/((self.lamb+1)*self.R)))*\
                                        ( (norm**self.lamb)/(self.R**self.lamb) )*event[x_map,y_map]
                    dist = (1 - (self.lamb*norm)/((self.lamb+1)*self.R))
                    dist = dist if dist >= 0 else 0
                    delta_V_t += (1 - (((x_p)@self.perspective.T))/norm)\
                                    *dist*((norm**self.lamb)/(self.R**self.lamb))\
                                        *event[x_map,y_map]
                    x_center += x*(((norm**self.lamb)*event[x_map,y_map] )/(self.R**self.lamb))
                    y_center += y*(((norm**self.lamb)*event[x_map,y_map] )/(self.R**self.lamb))
                
                v_V = v_V_t/mu_V
                delta_V = delta_V_t/mu_V
                delta_V = delta_V if delta_V > 0 else 1e-10
                alpha_v = acos(1-sqrt(delta_V))
                alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi
                
                centroid = np.array([x_center/mu_V, y_center/mu_V])
                translational_force += self.Kp*(self.norm(centroid - self.pos) - self.R*cos(self.alpha))\
                                                *self.perspective
                rotational_force += self.Kv*(np.eye(2) - np.dot(self.perspective[:,None],\
                                                self.perspective[None,:]))  @  (v_V.reshape(2,1))
                zoom += -self.Ka*(self.alpha - alpha_v)

            # control law for minimizing overlap area
            if len(self.overlap[0]) > 0:
                mu_V_o = 0
                v_V_t_o = np.array([0, 0], dtype=np.float64)
                delta_V_t_o = 0
                x_center_o = self.pos[0]
                y_center_o = self.pos[1]

                for i in range(len(self.overlap[0])):
                    x_map = self.overlap[1][i]
                    y_map = self.overlap[0][i]
                    x, y = x_map*self.grid_size[0], y_map*self.grid_size[1]
                    x_p = np.array([x,y]) - self.pos
                    norm = self.norm(x_p)
                    mu_V_o += ((norm**self.lamb)*event[x_map,y_map] )/(self.R**self.lamb)
                    v_V_t_o += ((x_p)/norm)*(cos(self.alpha) - \
                                    ( ( self.lamb*norm )/((self.lamb+1)*self.R)))*\
                                        ( (norm**self.lamb)/(self.R**self.lamb) )*event[x_map,y_map]
                    dist = (1 - (self.lamb*norm)/((self.lamb+1)*self.R))
                    dist = dist if dist >= 0 else 0
                    delta_V_t_o += (1 - (((x_p)@self.perspective.T))/norm)\
                                    *dist*((norm**self.lamb)/(self.R**self.lamb))\
                                        *event[x_map,y_map]
                    x_center_o += x*(((norm**self.lamb)*event[x_map,y_map] )/(self.R**self.lamb))
                    y_center_o += y*(((norm**self.lamb)*event[x_map,y_map] )/(self.R**self.lamb))

                v_V_o = v_V_t_o/mu_V_o
                delta_V_o = delta_V_t_o/mu_V_o
                delta_V_o = delta_V_o if delta_V_o > 0 else 1e-10
                alpha_v_o = acos(1-sqrt(delta_V_o))
                alpha_v_o = alpha_v_o if alpha_v_o > 5/180*np.pi else 5/180*np.pi

                rotational_force -= self.Kvo*(np.eye(2) - np.dot(self.perspective[:,None],\
                                                self.perspective[None,:]))@(v_V_o.reshape(2,1))
                zoom -= -self.Kao*(self.alpha - alpha_v_o)

            translational_force = translational_force
            return [translational_force, np.asarray([rotational_force[0][0], rotational_force[1][0]])\
                        , zoom]

        def PerspectiveQuality(self, x, y):
            x_p = np.array([x,y], dtype=object) - self.pos
            return (np.matmul(x_p,self.perspective.transpose())/np.linalg.norm(x_p)\
                - np.cos(self.alpha))/(1 - np.cos(self.alpha))

        def ResolutionQuality(self, x, y):
            x_p = np.array([x, y], dtype=object) - self.pos
            return (((np.linalg.norm(x_p)**self.lamb)*(self.R*np.cos(self.alpha)\
                 - self.lamb*( np.linalg.norm(x_p) - self.R*np.cos(self.alpha)) ))\
                    /(self.R**(self.lamb+1)))    

        def ComputeFoV(self):
            range_max = (self.lamb + 1)/(self.lamb)*self.R
            quality_map = None
            for y_map in range(max(int((self.pos[1] - range_max)/self.grid_size[1]), 0),\
                                min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[1])):
                x_map = np.arange(max(int((self.pos[0] - range_max)/self.grid_size[0]), 0),\
                                min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0]))
                q_per = self.PerspectiveQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
                q_res = self.ResolutionQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
                quality = np.where((q_per > 0) & (q_res > 0), q_per*q_res, 0)
                if quality_map is None:
                    quality_map = quality
                else:
                    quality_map = np.vstack((quality_map, quality))

            self.FoV[max(int((self.pos[1] - range_max)/self.grid_size[1]), 0):\
                        min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[0]),\
                            max(int((self.pos[0] - range_max)/self.grid_size[0]), 0):\
                                min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0])]\
                                    = quality_map
            self.polygon_FOV()

        def UpdateVoronoi(self, FoVs):
            if self.stage == 1:
                self.event = self.event_density(self.event, self.targets, self.grid_size) 
                self.event_plt = ((self.event - self.event.min()) * (1/(self.event.max() - self.event.min()) * 255)).astype('uint8')
            elif self.stage == 2:
                self.event = self.event_density(self.event, [self.targets[self.target_assigned]], self.grid_size) 
                self.event_plt = ((self.event - self.event.min()) * (1/(self.event.max() - self.event.min()) * 255)).astype('uint8')
            
            quality_map = self.FoV
            for FoV in FoVs:
                quality_map = np.where((quality_map > FoV), quality_map, 0)

            self.quality = np.sum(self.FoV*np.transpose(self.event))
            self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
            self.overlap = np.array(np.where((quality_map == 0) & (self.FoV != 0)))
            self.map_plt = np.array(np.where(quality_map != 0, self.id + 1, 0))

        def norm(self, arr):
            sum = 0
            for i in range(len(arr)):
                sum += arr[i]**2
            return sqrt(sum)

        def event_density(self, event, target, grid_size):
            x = np.arange(event.shape[0])*grid_size[0]
            for y_map in range(0, event.shape[1]):
                y = y_map*grid_size[1]
                density = 0
                for i in range(len(target)):
                    density += target[i][2]*np.exp(-target[i][1]*np.linalg.norm(np.array([x,y], dtype=object)\
                                    -np.array((target[i][0][1],target[i][0][0]))))
                event[:][y_map] = density

            return 0 + event 

    class Map():
        def __init__(self, map_size, grid_size):
            self.size = (np.array(map_size) / np.array(grid_size)).astype(np.int64)
            self.grid_size = grid_size
            self.window_size = np.array(self.size)*4
            self.display = pygame.display.set_mode(self.window_size)
            self.display.fill((0,0,0))
            self.blockSize = int(self.window_size[0]/self.size[0]) #Set the size of the grid block
            for x in range(0, self.window_size[0], self.blockSize):
                for y in range(0, self.window_size[1], self.blockSize):
                    rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                    pygame.draw.rect(self.display, (125,125,125), rect, 1)
            pygame.display.update()

        def Update(self, map_plt, cameras, event, targets, quality):
            x_map = 0
            for x in range(0, self.window_size[0], self.blockSize):
                y_map = 0
                for y in range(0, self.window_size[1], self.blockSize):
                    dense = event[x_map][y_map]
                    w = 0.6
                    id = int(map_plt[y_map][x_map])
                    if id == -1:
                        gray = (1-w)*125 + w*dense
                        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
                        pygame.draw.rect(self.display, (gray, gray, gray), rect, 0)
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
                center = camera.pos/self.grid_size*self.blockSize
                font = pygame.font.Font('freesansbold.ttf', 16)
                stage = " > Free" if camera.stage == 1 else ""
                text = font.render(str(camera.id)+" > "+ str(camera.target_assigned)+stage, \
                                    True, color)
                textRect = text.get_rect()
                textRect.center = (center[0], center[1] - 20) if camera.perspective[1] > 0 \
                                    else (center[0], center[1] + 20)
                R = camera.R*cos(camera.alpha)/self.grid_size[0]*self.blockSize
                self.display.blit(text, textRect)
                pygame.draw.line(self.display, color, center, center + camera.perspective*R, 3)
                pygame.draw.circle(self.display, color, camera.pos/self.grid_size*self.blockSize, 10)

                color = (camera.color[0]*0.5, camera.color[1]*0.5, camera.color[2]*0.5)
                pygame.draw.polygon(self.display, color, [camera.pos/self.grid_size*self.blockSize, \
                                                            camera.ltop/self.grid_size*self.blockSize, \
                                                            camera.top/self.grid_size*self.blockSize, \
                                                            camera.rtop/self.grid_size*self.blockSize], 2)
            for target in targets:
                pos = np.asarray(target[0])/self.grid_size*self.blockSize
                pygame.draw.circle(self.display, (0,0,0), pos, 4)
                pygame.draw.line(self.display, (0,0,0), pos, pos + target[3].reshape(1, 2)[0]/2\
                                    /self.grid_size*self.blockSize, 2)

            pygame.draw.rect(self.display, (0, 0, 0), (0, 0, map_size[0]/grid_size[0]*self.blockSize, \
                                                        map_size[1]/grid_size[1]*self.blockSize), width = 3)
            font = pygame.font.Font('freesansbold.ttf', 35)
            text = font.render(str(np.round(quality,2)),True, (100,175,255))
            textRect = text.get_rect()
            textRect.center = (100, 50)
            self.display.blit(text, textRect)
            pygame.display.flip()

def norm(arr):
        sum = 0
        for i in range(len(arr)):
            sum += arr[i]**2
        return sqrt(sum)

def event_density(event, target, grid_size):
    x = np.arange(event.shape[0])*grid_size[0]
    for y_map in range(0, event.shape[1]):
        y = y_map*grid_size[1]
        density = 0
        for i in range(len(target)):
            density += target[i][2]*np.exp(-target[i][1]*np.linalg.norm(np.array([x,y], dtype=object)\
                            -np.array((target[i][0][1],target[i][0][0]))))
        event[:][y_map] = density

    return 0 + event 

def dynamicTarget(x, y, v, res):
    turn = np.random.randint(-30, 30)/180*np.pi
    # u = (np.round(float(np.clip(dx/2 + x, 0, 24)),1),
    #          np.round(float(np.clip(dy/2 + y, 0, 24)),1))
    rot = np.array([[cos(turn), -sin(turn)],
                    [sin(turn), cos(turn)]])
    v = rot@v.reshape(2,1)
    vx = v[0] if v[0]*res + x > 0 and v[0]*res + x < 24 else -v[0]
    vy = v[1] if v[1]*res + y > 0 and v[1]*res + y < 24 else -v[1]
    return (x,y), np.asarray([[0],[0]])
    # return (np.round(float(np.clip(v[0]*res + x, 0, 24)),1), \
    #         np.round(float(np.clip(v[1]*res + y, 0, 24)),1)), np.round(np.array([[vx],[vy]]), len(str(res).split(".")[1]))

def randomUnitVector():
    v = np.asarray([np.random.normal() for i in range(2)])
    return v/norm(v)/2

if __name__ == "__main__":
    pygame.init()

    map_size = (24, 24)    
    grid_size = (0.1, 0.1)

    cameras = []
    camera0 = { 'id'            :  0,
                'position'      :  np.array([2.,3.]),
                'perspective'   :  np.array([0.9,1]),
                'AngleofView'   :  10,
                'range_limit'   :  3,
                'lambda'        :  2,
                'color'         : (200, 0, 0)}

    cameras.append(camera0)

    camera1 = { 'id'            :  1,
                'position'      :  np.array([3.,4.]),
                'perspective'   :  np.array([0.7,1]),
                'AngleofView'   :  10,
                'range_limit'   :  3,
                'lambda'        :  2,
                'color'         : (0, 200, 0)}

    cameras.append(camera1)

    camera2 = { 'id'            :  2,
                'position'      :  np.array([4.,5.]),
                'perspective'   :  np.array([0.7,1]),
                'AngleofView'   :  10,
                'range_limit'   :  3,
                'lambda'        :  2,
                'color'         : (150, 100, 200)}

    cameras.append(camera2)

    # camera3 = { 'id'            :  3,
    #             'position'      :  np.array([1.3, 4.7]),
    #             'perspective'   :  np.array([0,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  3,
    #             'lambda'        :  2,
    #             'color'         : (0, 100, 200)}

    # cameras.append(camera3)

    # camera4 = { 'id'            :  4,
    #             'position'      :  np.array([3.6, 5.]),
    #             'perspective'   :  np.array([0,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  3,
    #             'lambda'        :  2,
    #             'color'         : (175, 150, 0)}

    # cameras.append(camera4)

    # camera5 = { 'id'            :  5,
    #             'position'      :  np.array([1.1,2.4]),
    #             'perspective'   :  np.array([0,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  3,
    #             'lambda'        :  2,
    #             'color'         : (125, 150, 40)}

    # cameras.append(camera5)

    # camera6 = { 'id'            :  6,
    #             'position'      :  np.array([4.1,5.3]),
    #             'perspective'   :  np.array([0,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  3,
    #             'lambda'        :  2,
    #             'color'         : (75, 150, 0)}

    # cameras.append(camera6)

    # camera7 = { 'id'            :  7,
    #             'position'      :  np.array([4.6,5.1]),
    #             'perspective'   :  np.array([0,1]),
    #             'AngleofView'   :  10,
    #             'range_limit'   :  3,
    #             'lambda'        :  2,
    #             'color'         : (175, 50, 0)}

    # cameras.append(camera7)

    size = (np.array(map_size) / np.array(grid_size)).astype(np.int64)
    event = np.zeros((size[0], size[1]))

    target = [[(5, 5), 3, 10, randomUnitVector()], \
                [(12.5, 12.5), 3, 10,randomUnitVector()], \
                    [(17, 12.5), 3, 10, randomUnitVector()]] #target, certainty, weight, velocity
    event1 = event_density(event, target, grid_size)    
    event_plt1 = ((event1 - event1.min()) * (1/(event1.max() - event1.min()) * 255)).astype('uint8')

    voronoi = Voronoi2D(map_size, grid_size, len(cameras), cameras, np.ones(size), np.ones(size), target)

    Done = False
    start = time()
    cnt = [0 for i in range(len(target))]    
    while not Done:
        #last = time()
        for op in pygame.event.get():
            if op.type == pygame.QUIT:
                Done = True
        event = np.zeros((size[0], size[1]))
        for i in range(len(target)):
            if i == np.argmax(cnt) and cnt[i] != 0:
                target[i][2] -= 1
                target[i][2] = np.clip(target[i][2], 1e-10, 100)
            elif i == np.argmin(cnt) or cnt[i] == 0:
                target[i][2] += 1
                target[i][2] = np.clip(target[i][2], 1e-10, 100)
        if cnt[0] == 1 and cnt[1] == 1 and cnt[2] == 1:
            print(cnt)
            target[0][2] = 10
            target[1][2] = 10
            target[2][2] = 10

        target_dyn = [dynamicTarget(target[i][0][0], target[i][0][1], target[i][3], grid_size[0])\
                         for i in range(len(target))]
        target = [[target_dyn[0][0], 3, target[0][2], target_dyn[0][1]],
                    [target_dyn[1][0], 3, target[1][2], target_dyn[1][1]],
                        [target_dyn[2][0], 3, target[2][2], target_dyn[2][1]]]
        event_plt1 = ((event1 - event1.min()) * (1/(event1.max() - event1.min()) * 255)).astype('uint8')
        cnt = voronoi.Update(event_plt=event_plt1, target=target, step = 0.1)

        # print("===================", cnt, "====================")
        #print(time() - last)
        #print("Total Runtime: ", np.round(time() - start,2), "s")
        #print("Single Runtime: ", np.round(((time() - start)/len(cameras)),2), "s")
    
    pygame.quit()
