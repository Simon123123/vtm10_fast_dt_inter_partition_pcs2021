import numpy as np
import pandas as pd 
import os


list_files_cost = [] 
list_files_features = []

#Path to the split_cost_yuvname.csv files

path_split_cost = ""

#Path to the split_features_yuvname.csv files

path_split_feature = ""


for f in sorted(os.listdir(path_split_cost)):
    if f.startswith("split_cost"):
        list_files_cost.append(os.path.join(path_split_cost, f))


for f in sorted(os.listdir(path_split_feature)):
    if f.startswith("split_features"):
        list_files_features.append(os.path.join(path_split_feature, f))


frames_272_544 = range(3, 64, 3)

frames_others = [8, 16, 28, 42, 49, 57]


for f_c, f_f in zip(list_files_cost, list_files_features):

    cost = pd.read_csv(f_c, delimiter=';', header = None, keep_default_na=False).to_numpy()
    name_col = np.arange(53)
    features = pd.read_csv(f_f, delimiter=';', header = None, keep_default_na=False, names=name_col).to_numpy()

    frames = frames_272_544 if ("272" in f_c or "544" in f_c) else frames_others

    for fr in frames:

        file_name_c = os.path.split(f_c)[1].replace('.csv', '.npy')
        file_name_f = os.path.split(f_f)[1].replace('.csv', '.npy')

        np.save(r"/output_path/c_per_frame/f" + str(fr) + "_" + file_name_c, cost[cost[:, 0] == fr])

        np.save(r"/output_path/f_per_frame/f" + str(fr) + "_" + file_name_f, features[features[:, 0] == fr])







