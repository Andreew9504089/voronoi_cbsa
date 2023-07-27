import os 
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self):
        pass
    
    def SinglePlotCoop(self, paths, agent_id = -1):
        fig  = plt.figure()
        ax1 = fig.add_subplot()
        
        color_pool = {'coop': 'r', 'non-coop': 'g'}
        
        if agent_id != -1:
            for type  in paths.keys():
                df = pd.read_csv(paths[type]+type+"_"+str(agent_id)+".csv")
                col = str(agent_id)+"'s score"
                data = np.array((df['frame_id'].values, df[col].values))
                label = type
                ax1.plot(data[0][:], np.log10(data[1][:]), color = color_pool[type], label=label)
        
        else:
            for type in paths.keys():
                frames = []
                scores = {}
                max_len = 0
                
                files = sorted(glob.glob(os.path.join(paths[type], "*.csv")))
                for i, file in enumerate(files):
                    print(file)
                    df = pd.read_csv(file)
                    col = str(i)+"'s score"
                    data = np.array((df['frame_id'].values, df[col]))
                    scores[i] = (data[1][:])

                    # if len(data[0][:]) < len(frames) or i == 0:
                    #     frames = data[0][:]
                    
                    if len(data[0][:]) > max_len:
                        max_len = len(data[0][:])
                        frames = data[0][:]
                
                total_mat = []  
                for i in scores.keys():
                    tmp = np.squeeze(np.zeros((max_len, 1)))
                    tmp[0:len(scores[i][:])] = scores[i][:]
                    total_mat = tmp if i == 0 else np.dstack([total_mat, tmp])
                
                result = np.squeeze(np.sum(total_mat, axis=2))

                label = type
                ax1.plot(frames, result, color = color_pool[type], label=label)
                
        title = "Score of agent " + str(agent_id) if agent_id != -1 else "Total Score"
        ax1.set_xlabel("frame id")
        ax1.set_ylabel("utility")
        ax1.set_title(title)
        
        plt.legend() 
        plt.show()
    
    def MultiPlotCoop(self, paths):
        files = {'coop'     : sorted(glob.glob(os.path.join(paths['coop'], "*.csv"))),
                 'non-coop' : sorted(glob.glob(os.path.join(paths['non-coop'], "*.csv")))}
        
        agent_num = len(files['coop'])
        fig, ax  = plt.subplots(math.ceil(agent_num/2), 2, constrained_layout = True)
        color_pool = {'coop': 'r', 'non-coop': 'g'}
        
        for type in paths.keys():
            for i, file in enumerate(files[type]):
                df = pd.read_csv(file)
                col = str(i)+"'s score"
                data = np.array((df['frame_id'].values, df[col].values))
                
                label = type
                title = "Score of agent " + str(i)
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].plot(data[0][:], (data[1][:]), color = color_pool[type], label=label)
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].legend(loc="upper left")
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].set_xlabel("frame id")
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].set_ylabel("utility")
                ax[i % math.ceil(agent_num/2), i // math.ceil(agent_num/2)].set_title(title)
        
        plt.legend() 
        plt.show()
    
    def PlotTrajectory(self, paths):
        files = {'coop'     : sorted(glob.glob(os.path.join(paths['coop'], "*.csv"))),
                 'non-coop' : sorted(glob.glob(os.path.join(paths['non-coop'], "*.csv")))}
        
        fig, ax  = plt.subplots(1, 2, constrained_layout = True)
        color_pool = [np.array((255, 0, 0)), np.array((255, 128, 0)), np.array((255,255,0)),
                      np.array((0,255,0)), np.array((0,255,255)), np.array((0,0,255)),
                      np.array((178,102,255)), np.array((255,0,255)), np.array((13, 125, 143))]
        for j, type in enumerate(paths.keys()):
            for i, file in enumerate(files[type]):
                df = pd.read_csv(file)
                col_x = "pos_x"
                col_y = "pos_y"
                data = np.array((df[col_x].values, df[col_y].values))
                
                label = "agent " + str(i)
                ax[j].scatter(data[1][0], data[0][0], marker="s",s=50,color = color_pool[i]/255*0.5, label=label+" start", zorder=2)
                ax[j].scatter(data[1][-1], data[0][-1], marker="^",s=50, color = color_pool[i]/255*0.5, label=label+" end", zorder=2)
                ax[j].plot(data[1][:], (data[0][:]), color = color_pool[i]/255, label=label, zorder=1)
                
            
            title = "Trajectories of " + type
            ax[j].legend(loc="upper left")
            ax[j].set_xlabel("x")
            ax[j].set_ylabel("y")
            ax[j].set_title(title)
        
        plt.legend() 
        plt.show()
    
if __name__ == "__main__":
    eval = Evaluator()
    
    trial_id = '8'
    paths = {'coop'     : "/home/andrew/research_ws/src/voronoi_cbsa/result/"+trial_id+"/coop/", 
             'non-coop' : "/home/andrew/research_ws/src/voronoi_cbsa/result/"+trial_id+"/non-coop/"}

    eval.SinglePlotCoop(paths, -1)
    eval.MultiPlotCoop(paths)
    #eval.PlotTrajectory(paths)

    
    