import numpy as np
import os, sys
from multiprocessing import Process



def get_feature_per_frame(ind_f):


    list_files_horver = [] 
    list_files_qtmt = []


# path_feature_files represents the path of the output files after running script feature_ext.py

    path_feature_files = ""

    ind_str =  "_" + str(ind_f) + ".npy"

    for f in sorted(os.listdir(path_feature_files)):
        if f.startswith("horver_") and ind_str in f:
            list_files_horver.append(os.path.join(path_feature_files, f))


    for f in sorted(os.listdir(path_feature_files)):
        if f.startswith("qtmt_") and ind_str in f:
            list_files_qtmt.append(os.path.join(path_feature_files, f))


    np_hv = np.zeros((1, 54))

    np_qtmt = np.zeros((1, 43))



    for f_hv in list_files_horver:

        features = np.load(f_hv, allow_pickle=True)

        np_hv = np.vstack((np_hv, features))


    for f_qtmt in list_files_qtmt:

        features = np.load(f_qtmt, allow_pickle=True)

        np_qtmt = np.vstack((np_qtmt, features))



    f_16x16_qtmt = np_qtmt[(np_qtmt[:, 1:3] == np.array((16, 16)).reshape((1,2))).all(axis=1)]

    f_16x16_qtmt_f = f_16x16_qtmt[~((f_16x16_qtmt == 'inf') | (f_16x16_qtmt == 'nan')).any(axis = 1)]


    np.save(r"/output_path/qm_16x16_" + str(ind_f) + ".npy", f_16x16_qtmt_f)



    f_32x32_qtmt = np_qtmt[(np_qtmt[:, 1:3] == np.array((32, 32)).reshape((1,2))).all(axis=1)]

    f_32x32_qtmt_f = f_32x32_qtmt[~((f_32x32_qtmt == 'inf') | (f_32x32_qtmt == 'nan')).any(axis = 1)]

    np.save(r"/output_path/qm_32x32_" + str(ind_f) + ".npy", f_32x32_qtmt_f)





    f_64x64_qtmt = np_qtmt[(np_qtmt[:, 1:3] == np.array((64, 64)).reshape((1,2))).all(axis=1)]

    f_64x64_qtmt_f = f_64x64_qtmt[~((f_64x64_qtmt == 'inf') | (f_64x64_qtmt == 'nan')).any(axis = 1)]

    np.save(r"/output_path/qm_64x64_" + str(ind_f) + ".npy", f_64x64_qtmt_f)



    f_128x128_qtmt = np_qtmt[(np_qtmt[:, 1:3] == np.array((128, 128)).reshape((1,2))).all(axis=1)]

    f_128x128_qtmt_f = f_128x128_qtmt[~((f_128x128_qtmt == 'inf') | (f_128x128_qtmt == 'nan')).any(axis = 1)]

    np.save(r"/output_path/qm_128x128_" + str(ind_f) + ".npy", f_128x128_qtmt_f)


    # horver features

    horver = [(8,8), (8,16), (8,32), (8,64), (16,8), (16,16), (16,32), (16,64), (32,8), (32,16), (32,32), (32,64), (64,8), (64,16), (64,32), (64,64), (128,128)]



    for dim in horver:

        horver_f = np_hv[(np_hv[:, 1:3] == np.array(dim).reshape((1,2))).all(axis=1)]

        horver_ff = horver_f[~((horver_f == 'inf') | (horver_f == 'nan')).any(axis = 1)]

        np.save(os.path.join(r"/output_path", "hv_" + str(dim[0]) + "x" + str(dim[1]) + "_" + str(ind_f) + ".npy"), horver_ff)





p_list = []

f_ind = int(sys.argv[1])

start = 200 * f_ind

end = min(200 * (f_ind + 1), 1527)

for i in range(start, end):

    p = Process(target=get_feature_per_frame, args=(i,))
    p.start()
    p_list.append(p)

for p in p_list:
    p.join()