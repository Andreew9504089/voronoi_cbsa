import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    def __inti__(self):
        pass
    
    def SinglePlotCoop(self, paths, agent_id = -1):
        fig  = plt.figure(0)
        ax1 = fig.add_subplot()
        
        color_pool = {'coop': 'r', 'non-coop': 'g'}
        for type  in enumerate(paths.keys()):
            df = pd.read_csv(paths[type])
            data = np.array((df['frame_id'].values, df['pos_y'].values))
            label = type
            ax1.plot(data[0][:], data[1][:], color = color_pool[i], label=label)
        
        title = "Score of agent " + str(agent_id) if agent_id != -1 else "Total Score"
        ax1.set_xlim([-1, 13])
        ax1.set_ylim([-3, 3])
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_label(title)
        
        plt.legend() 
        plt.show()
    
    def MultiPLotCoop(self):
        pass

    def SinglePlotBalance(self):
        pass
    
    def MultiPlotBalance(self):
        