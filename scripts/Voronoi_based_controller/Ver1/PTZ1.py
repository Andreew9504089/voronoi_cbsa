from Voronoi2D import Voronoi2D
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray

class FoVHandler():
    def __init__(self, total_PTZs):
        self.FoVs = []
        self.FoVs_sub = []
        for i in range(total_PTZs):
            self.FoVs.append(None)
            if i != id:
                topic = "PTZ" + str(id) + "/FoV"
                self.FoVs_sub.append(rospy.Subscriber(topic), Float64MultiArray, self.FoVUpdate(i))
            else:
                self.FoVs_sub.appen(None)

    def FoVUpdate(self, id, data):
        self.FoVs[id] = np.array(data.data)

    def FoVFetch(self):
        return self.FoVs

    def FoVClear(self):
        for i in range(total_PTZs):
            self.FoVs[i] = None
        
        return self.FoVs

class eventHandler():
    def __init__(self):
        self.event = []
        self.sub = rospy.Subscriber("/event", Float64MultiArray, self.EventUpdate)

    def EventUpdate(self, data):
        self.event = np.array(data.data)
    
    def EventFetch(self):
        return self.event

rospy.init_node('PTZ1')

#map_size = rospy.get_param("/map_size")
#grid_size = rospy.get_param("/grid_size")

total_PTZs = 8
id = 0
rate = rospy.sleep(30)

FoV_pub = rospy.Publisher("PTZ1/FoV")
FoV_handler = FoVHandler(total_PTZs)
event_handler = eventHandler()

map_size = (10, 10)    
grid_size = (0.1, 0.1)

cameras = []
camera0 = { 'id'            :  0,
            'position'      :  np.array([0,0]),
            'perspective'   :  np.array([1,0]),
            'AngleofView'   :  10,
            'range_limit'   :  3,
            'lambda'        :  2,
            'color'         : (200, 0, 0)}

cameras.append(camera0)
ptz = Voronoi2D.PTZcamera(camera0, map_size, grid_size)

while not rospy.is_shutdown():
    rospy.spin()
    event = event_handler.EventFetch()
    FoVs = FoV_handler.FoVFetch()

    ptz.ComputeFoV()
    FoV_pub.publish(ptz.FoV)

    ptz.UpdateVoronoi(FoVs)
    ptz.UpdateState(event)

    FoV_handler.FoVClear()
    ptz.FoV = []
    ptz.voronoi = []
    