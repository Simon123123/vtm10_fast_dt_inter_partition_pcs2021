import numpy as np
import math, sys, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib




qm_sample_nums = [36981, 14399, 25120, 10899]



path_dataset = ""

qm_size = sys.argv[1]

name_file = "qm_" + str(qm_size) + "x" + str(qm_size) + ".npy" 

qm_f = np.load(path_dataset + name_file, allow_pickle=True)

qm_f = qm_f.astype(float)

qm_f = qm_f[~np.isnan(qm_f).any(axis=1), :]

qm_f = qm_f[~np.isinf(qm_f).any(axis=1), :]


ind = list(range(qm_f.shape[0]))

random.shuffle(ind)

num_sample = qm_sample_nums[int(math.log2(int(qm_size))) - 4]

qm_f = qm_f[ind, :][:num_sample, :]

print("Num of samples after calculation is:", qm_f.shape[0])






model = RandomForestClassifier(n_estimators=40, n_jobs=8, max_depth = 20, min_samples_split=100)

x_train, x_test, y_train, y_test = train_test_split(qm_f[:, 7:41], qm_f[:, 41:], test_size=0.2, random_state=42, shuffle=True)

np.save(r'/output_path/y_test' + name_file, y_test)

np.save(r'/output_path/x_test' + name_file, x_test)

model.fit(x_train, y_train[:, 0], y_train[:, 1])

joblib.dump(model, r'/output_path/' + name_file.replace('.npy', '.pkl'), compress = 0)

per_tree_pred = [tree.predict_proba(x_test) for tree in model.estimators_]

np.save(r'/output_path/pred_by_tree_' + name_file, np.concatenate(per_tree_pred, axis=0))

y_pred = model.predict(x_test)

print('Model accuracy score with 40 decision-trees for qm size {} is : {}'. format(qm_size, accuracy_score(y_test[:, 0], y_pred)))

