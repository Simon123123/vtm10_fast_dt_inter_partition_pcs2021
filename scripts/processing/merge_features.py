import numpy as np
import os


qm_sizes = [128, 64, 32, 16]

# path_feature_files is the folder containing the output files of sep_cu_sizes.py

path_feature_files = ""


for s in qm_sizes:

    list_files_qtmt = []

    size_str = str(s) + "x" + str(s)

    for f in sorted(os.listdir(path_feature_files)):
        if f.startswith("qm_") and size_str in f:
            list_files_qtmt.append(os.path.join(path_feature_files, f))

    np_qm = np.zeros((1, 43))

    for f_qm in list_files_qtmt:

        features = np.load(f_qm, allow_pickle=True)

        np_qm = np.concatenate((np_qm, features), axis = 0)


    num_class_0 = sum(np_qm[:, 41] == 0)

    num_class_1 = sum(np_qm[:, 41] == 1)

    dim_f = min(num_class_0, num_class_1)
    
    ind_f = np.arange(dim_f)

    class_0 = np_qm[np_qm[:, 41] == 0]

    class_1 = np_qm[np_qm[:, 41] == 1]

    feature_eq = np.concatenate((class_0[ind_f, :], class_1[ind_f, :]), axis = 0) 

    np.save(r"/output_path/qm_" + size_str + ".npy", feature_eq)    



horver = [(8,8), (8,16), (8,32), (8,64), (16,8), (16,16), (16,32), (16,64), (32,8), (32,16), (32,32), (32,64), (64,8), (64,16), (64,32), (64,64), (128,128)]

for s in horver:

    list_files_hv = []

    size_str = str(s[0]) + "x" + str(s[1])

    for f in sorted(os.listdir(path_feature_files)):
        if f.startswith("hv_") and size_str in f:
            list_files_hv.append(os.path.join(path_feature_files, f))

    np_hv = np.zeros((1, 54))

    for f_hv in list_files_hv:

        features = np.load(f_hv, allow_pickle=True)

        np_hv = np.concatenate((np_hv, features), axis = 0)

    num_class_0 = sum(np_hv[:, 52] == 0)

    num_class_1 = sum(np_hv[:, 52] == 1)

    dim_f = min(num_class_0, num_class_1)
    
    ind_f = np.arange(dim_f)

    class_0 = np_hv[np_hv[:, 52] == 0]

    class_1 = np_hv[np_hv[:, 52] == 1]

    feature_eq = np.concatenate((class_0[ind_f, :], class_1[ind_f, :]), axis = 0) 

    np.save(r"/output_path/hv_" + size_str + ".npy", feature_eq)



