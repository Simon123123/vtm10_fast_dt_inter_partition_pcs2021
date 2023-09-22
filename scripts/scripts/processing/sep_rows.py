import numpy as np
import os


list_files_cost = [] 
list_files_features = []

for f in sorted(os.listdir(r"/output_path_for_4k/c_per_frame")):
    if f.startswith("f"):
        list_files_cost.append(os.path.join(r"/output_path_for_4k/c_per_frame", f))


for f in sorted(os.listdir(r"/output_path_for_4k/f_per_frame")):
    if f.startswith("f"):
        list_files_features.append(os.path.join(r"/output_path_for_4k/f_per_frame", f))



for f_f, f_c in zip(list_files_features, list_files_cost):


    features = np.load(f_f, allow_pickle=True)
    
    cost = np.load(f_c, allow_pickle=True)

    for row in range(17):

        file_name_f = os.path.split(f_f)[1]

        file_name_c = os.path.split(f_c)[1]


        np.save(r"/output_path_for_4k/f_per_row/r" + str(row) + "_" + file_name_f, features[np.logical_and(features[:, 4] >= row * 128, features[:, 4] < (row + 1) * 128) ])
        
        np.save(r"/output_path_for_4k/c_per_row/r" + str(row) + "_" + file_name_c, cost[np.logical_and(cost[:, 4] >= row * 128, cost[:, 4] < (row + 1) * 128) ])

