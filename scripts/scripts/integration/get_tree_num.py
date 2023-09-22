import numpy as np

#path_dt_eval is the path to the output folder of script eval_rf_models.py

path_dt_eval = ""

set_hv = [(8,8), (8,16), (8,32), (8,64), (16,8), (16,16), (16,32), (16,64), (32,8), (32,16), (32,32), (32,64), (64,8), (64,16), (64,32), (64,64), (128,128)]

for dim in set_hv:

    dim_0 = int(dim[0])
    dim_1 = int(dim[1])

    set_tree = np.load(path_dt_eval + r"/eval_trees/set_hv_" + str(dim_0) + "_" + str(dim_1) + ".npy")

    acc_tree = np.load(path_dt_eval + r"/eval_trees/acc_hv_" + str(dim_0) + "_" + str(dim_1) + ".npy")

    ind_best_acc = np.argmax(acc_tree[10:]) + 10

    print("For hv rf with dim {} x dim {} the best trees is at {} with acc {}".format(dim_0, dim_1, ind_best_acc, acc_tree[ind_best_acc]))
    print("The best tree indexs are: ")
    for i in range(ind_best_acc):
        print(str(set_tree[i]) +  ',', end=' ')
    print(' ')



qm_size = [128, 64, 32, 16]


for dim in qm_size:

    dim_qm = int(dim)

    set_tree = np.load(path_dt_eval + r"/eval_trees/set_qm_" + str(dim_qm) + "_" + str(dim_qm) + ".npy")

    acc_tree = np.load(path_dt_eval + r"/eval_trees/acc_qm_" + str(dim_qm) + "_" + str(dim_qm) + ".npy")

    ind_best_acc = np.argmax(acc_tree[10:]) + 10

    print("For qm rf with dim {} x dim {} the best trees is at {} with acc {}".format(dim_qm, dim_qm, ind_best_acc, acc_tree[ind_best_acc]))
    print("The best tree indexs are: ")
    for i in range(ind_best_acc):
        print(str(set_tree[i]) +  ',', end=' ')

    print(' ')
