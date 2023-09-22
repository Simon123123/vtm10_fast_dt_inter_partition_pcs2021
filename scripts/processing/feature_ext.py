import numpy as np
import os, sys
from multiprocessing import Process



#This script use multiprocessing to process 50 tasks on one CPU / core


def get_feature_per_frame(ind_f):


    list_files_cost = [] 
    list_files_features = []


    #Path to the cost files seperated by rows(for 4k encodings) or by frames(for encodings on other resolutions). 

    path_split_cost_divided = ""

    #Path to the feature files seperated by rows(for 4k encodings) or by frames(for encodings on other resolutions).

    path_split_feature_divided = ""



    for f in sorted(os.listdir(path_split_cost_divided)):
        list_files_cost.append(os.path.join(path_split_cost_divided, f))


    for f in sorted(os.listdir(path_split_feature_divided)):
        list_files_features.append(os.path.join(path_split_feature_divided, f))


    f_c = list_files_cost[ind_f]

    f_f = list_files_features[ind_f]


    cost = np.load(f_c, allow_pickle=True)
    cost_decision = np.zeros((1,11))

    for r in cost:
        ind = np.where((cost_decision[:, :6]==r[:6]).all(axis=1))
        if ind[0].size:
            if (cost_decision[ind, int(r[6]-1)] > r[7]) or (cost_decision[ind, int(r[6]-1)] == 0):                
                cost_decision[ind, int(r[6]-1)] = r[7]
        else:
            row = np.zeros((11,))
            row[:6,] = r[:6]
            row[int(r[6]-1),] = r[7]
            cost_decision = np.vstack((cost_decision, row))



    features = np.load(f_f, allow_pickle=True)

    qtmt_f = features[(features[:, 6] == 0), :][:, :41]

    bhbv_f = features[(features[:, 6] == 1), :][:, :52]

    decision_qtmt = np.full((qtmt_f.shape[0], 2), -1)

    decision_horver = np.full((bhbv_f.shape[0], 2), -1)

    for r, ind_r in zip(qtmt_f, range(qtmt_f.shape[0])):
        
        ind = np.nonzero((cost_decision[:, :6] == r[:6]).all(axis = 1))
        
        if len(ind[0]) > 1:
            raise Exception("The length is not normal")
        elif len(ind[0]) == 1:
            if (cost_decision[ind, 7:] > 0).sum() <= 1:
                continue
            
            cost_qt = cost_decision[ind[0], 6]
            costs_mt = cost_decision[ind[0], 7:]
            min_mt = min(costs_mt[costs_mt > 0], default=-1)
            
            if cost_qt == 0 or min_mt == -1:
                continue
            elif cost_qt > min_mt:
                decision_qtmt[ind_r, 0] = 1
                decision_qtmt[ind_r, 1] = cost_qt - min_mt
            elif cost_qt < min_mt:
                decision_qtmt[ind_r, 0] = 0
                decision_qtmt[ind_r, 1] = min_mt - cost_qt




                            
    for r, ind_r in zip(bhbv_f, range(bhbv_f.shape[0])):
        
        ind = np.nonzero((cost_decision[:, :6] == r[:6]).all(axis = 1))
        
        if len(ind[0]) > 1:
            raise Exception("The length is not normal")
        elif len(ind[0]) == 1:
            if (cost_decision[ind, 7:] > 0).sum() <= 1:
                continue
            
            costs_hor = cost_decision[ind[0], 7::2]
            costs_ver = cost_decision[ind[0], 8::2]
            min_hor = min(costs_hor[costs_hor > 0], default=-1)
            min_ver = min(costs_ver[costs_ver > 0], default=-1)        
            
            if min_hor == -1 or min_ver == -1:
                continue
            elif min_hor < min_ver:
                decision_horver[ind_r, 0] = 0
                decision_horver[ind_r, 1] = min_ver - min_hor
            elif min_ver < min_hor:
                decision_horver[ind_r, 0] = 1
                decision_horver[ind_r, 1] = min_hor - min_ver            
                

        
    qtmt_f = np.hstack((qtmt_f, decision_qtmt))

    qtmt_f = qtmt_f[qtmt_f[:, -2] != -1]



    bhbv_f = np.hstack((bhbv_f, decision_horver))

    bhbv_f = bhbv_f[bhbv_f[:, -2] != -1]


    np.save(r"/output_path/qtmt_features_" + str(ind_f) + ".npy", qtmt_f)
    np.save(r"/output_path/horver_features_" + str(ind_f) + ".npy", bhbv_f)






p_list = []

f_ind = int(sys.argv[1])

for i in range(f_ind*50, (f_ind+1) * 50):

    p = Process(target=get_feature_per_frame, args=(i,))
    p.start()
    p_list.append(p)

for p in p_list:
    p.join()