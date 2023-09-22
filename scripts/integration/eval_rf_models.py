import numpy as np


# path_prediction is the path to the output folder of rf_train_models_hv.py and rf_train_models_qm.py

path_prediction = ""



set_hv = [(8,8), (8,16), (8,32), (8,64), (16,8), (16,16), (16,32), (16,64), (32,8), (32,16), (32,32), (32,64), (64,8), (64,16), (64,32), (64,64), (128,128)]

for dim in set_hv:

    dim_0 = int(dim[0])
    dim_1 = int(dim[1])

    pred_test = np.load(path_prediction + r"/pred_by_tree_hv_" + str(dim_0) + "_" + str(dim_1) + ".npy")

    res = np.load(path_prediction + r"/y_testhv_" + str(dim_0) + "_" + str(dim_1) + ".npy")

    size_pred = res.shape[0]

    tree_acc = []
    list_of_uneval = list(range(40))
    set_of_eval = []
    selected_proba = np.zeros((size_pred, 2))

    for i in range(40):

        acc_j = 0
        selected_tree = 0
        
        for j in list_of_uneval:
            
            new_proba = selected_proba + pred_test[j*size_pred : (j+1)*size_pred, :] 
            new_res = np.argmax(new_proba, axis = -1)
            new_acc = np.sum(new_res == res[:, 0])/size_pred
            
            if new_acc > acc_j:
                acc_j = new_acc
                selected_tree = j
        
        tree_acc.append(acc_j)
        set_of_eval.append(selected_tree)
        selected_proba = selected_proba + pred_test[selected_tree*size_pred : (selected_tree + 1)*size_pred, :] 
        list_of_uneval.remove(selected_tree)

    np.save(r"/output_path/eval_trees/" + "acc_hv_" + str(dim_0) + "_" + str(dim_1) + ".npy", tree_acc)
    np.save(r"/output_path/eval_trees/" + "set_hv_" + str(dim_0) + "_" + str(dim_1) + ".npy", set_of_eval)



qm_size = [128, 64, 32, 16]


for dim in qm_size:

    dim_qm = int(dim)

    pred_test = np.load(path_prediction + r"/pred_by_tree_qm_" + str(dim_qm) + "_" + str(dim_qm) + ".npy")

    res = np.load(path_prediction + r"/y_testqm_" + str(dim_qm) + "_" + str(dim_qm) + ".npy")

    size_pred = res.shape[0]

    tree_acc = []
    list_of_uneval = list(range(40))
    set_of_eval = []
    selected_proba = np.zeros((size_pred, 2))

    for i in range(40):

        acc_j = 0
        selected_tree = 0
        
        for j in list_of_uneval:
            
            new_proba = selected_proba + pred_test[j*size_pred : (j+1)*size_pred, :] 
            new_res = np.argmax(new_proba, axis = -1)
            new_acc = np.sum(new_res == res[:, 0])/size_pred
            
            if new_acc > acc_j:
                acc_j = new_acc
                selected_tree = j
        
        tree_acc.append(acc_j)
        set_of_eval.append(selected_tree)
        selected_proba = selected_proba + pred_test[selected_tree*size_pred : (selected_tree + 1)*size_pred, :] 
        list_of_uneval.remove(selected_tree)

    np.save(r"/output_path/eval_trees/" + "acc_qm_" + str(dim_qm) + "_" + str(dim_qm) + ".npy", tree_acc)
    np.save(r"/output_path/eval_trees/" + "set_qm_" + str(dim_qm) + "_" + str(dim_qm) + ".npy", set_of_eval)