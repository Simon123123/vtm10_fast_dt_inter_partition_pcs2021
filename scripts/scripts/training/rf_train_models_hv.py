import numpy as np
import os, sys, random, math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


h_size = sys.argv[1]
w_size = sys.argv[2]


hv_sample_nums = [[171506, 79935, 53239, 120340, 0],
                  [108806, 119899, 90806, 86392, 0],
                  [67925, 83998, 46213, 40698, 0],
                  [120340, 86392, 33719, 33325, 0],
                  [0, 0, 0, 0, 51396],               
                ]     


path_dataset = ""



name_file = "hv_" + str(h_size) + "_" + str(w_size) + ".npy" 

hv_f = np.load(path_dataset + name_file, allow_pickle=True)

hv_f = hv_f.astype(float)

hv_f = hv_f[~np.isnan(hv_f).any(axis=1), :]

hv_f = hv_f[~np.isinf(hv_f).any(axis=1), :]

ind = list(range(hv_f.shape[0]))

random.shuffle(ind)

num_sample = hv_sample_nums[int(math.log2(int(h_size))) - 3][int(math.log2(int(w_size))) - 3]

hv_f = hv_f[ind, :][:num_sample, :]


model = RandomForestClassifier(n_estimators=40, n_jobs=8, max_depth = 20, min_samples_split=100)

x_train, x_test, y_train, y_test = train_test_split(hv_f[:, 7:52], hv_f[:, 52:], test_size=0.2, random_state=42, shuffle=True)

np.save(r'/output_path/y_test' + name_file, y_test)

np.save(r'/output_path/x_test' + name_file, x_test)

model.fit(x_train, y_train[:, 0], y_train[:, 1])

joblib.dump(model, r'/output_path/' + name_file.replace('.npy', '.pkl'), compress=0)  

per_tree_pred = [tree.predict_proba(x_test) for tree in model.estimators_]

np.save(r'/output_path/pred_by_tree_' + name_file, np.concatenate(per_tree_pred, axis=0))

y_pred = model.predict(x_test)

print('Model accuracy score with 40 decision-trees for hv size {} x {} is : {}'. format(h_size, w_size, accuracy_score(y_test[:, 0], y_pred)))

