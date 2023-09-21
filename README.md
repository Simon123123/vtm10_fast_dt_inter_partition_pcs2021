Fast Decsion Tree based Inter Partitioning of VVC (PCS 2021)
============================================================

This is the reproduction of paper **Fast Versatile Video Coding using Specialised Decsion Trees** [1] by Gosala Kulupana et al. published in Picture Coding Symposium 2021. Many thanks to Mr kulupana for sharing the source code of his implementation in VTM8 with me. This serves as the stat of art comparaison in our paper 

**CNN-based Prediction of Partition Path for VVC Fast Inter Partitioning Using Motion Fields** [2] currently under review of IEEE Transaction of Image Processing. Comparing to original paper, we have generated a new dataset and trained the decision tree classifers.

Then the proposed method and trained decision trees are integrated into VTM10. With the approvement of Kulupana, we share the dataset, trained decision tree, and all the related code in this repository. Python scripts for processing the raw data and training && pruning the decision tree are included in folder scripts.



Build instructions
------------------

**It is generally suggested to build 64-bit binaries for VTM software**. We have built the software on Windows (MS Visual Studio) and Linux (make).  




**Windows Visual Studio 64 Bit:**

Use the proper generator string for generating Visual Studio files, e.g. for VS 2019:

```bash
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
```

Then open the generated solution file in MS Visual Studio.

For VS 2017 use "Visual Studio 15 2017 Win64", for VS 2019 use "Visual Studio 16 2019".

Visual Studio 2019 also allows you to open the CMake directory directly. Choose "File->Open->CMake" for this option.

For the release build of this project in Visual Studio, we should set the EncoderApp as Startup Project. Beforing building it, 
right click on the EncoderLib and set the "Treat Warnings As Error" to No in "Property->C/C++/General". Then add the compile option 
/bigobj in "Property->C/C++/Command Line" for EncoderLib. 
 

**Linux**

For generating Linux Release Makefile:
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```
For generating Linux Debug Makefile:
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

Then type
```bash
make -j
```

For more details, refer to the CMake documentation: https://cmake.org/cmake/help/latest/



Dataset Generation
------------------

In original paper, the dataset is built on a part of Common Test Condition (CTC) sequences while the rest of sequences is used for evaluating the method. In our implementation, we choose to generate the dataset on BVI-DVC database [3] and Youtube UVG database [4] to evaluate the performance on full CTC. All sequences in     
these databases are encoded. Data are selectively collected on some inter frames of encoded sequences. More precisely, we collect data on one frame every three frames for resolution 960x544 and 480x272. For other resolutions, we specifically collect the frames with POC equal to 8, 16, 28, 42, 49. To adjust the frames to
for the collection of data, please modify the "frame_collect" variable defined in line 1551 and 2905 in file EncModeCtrl.cpp. The collected features are stored in two generated csv files, namely "split_cost_yuvname.csv" and "split_features_yuvname.csv". For generating the dataset, please run the encoding with macro COLLECT_DATASET
activated in line 68 of file TypeDef.h. After obtaining the csv files, several python scripts are needed to process the collected data.


1. Firstly, the script sep_frames.py is executed to seperate collected data by frames. For 4k resolution, the frame level data is then split by rows after the execution of script sep_rows.py. The split of large data file facilates the multi-processing of files.

2. Then we execute the script feature_ext.py to extract the features for QT/MT decision and for Hor/Ver decision respectively for each data file.

3. Finally the feature files are merged per CU size and per decision. We have randomly selected equal number of samples for each decision.

The scripts for above steps are provided in folder script/processing. Finally, a dataset containing 21 numpy files is generated. If you seek for details about the format of data in this datase, please refer to the code and original paper. 

Here is our dataset: https://1drv.ms/f/s!Aoi4nbmFu71Hgn0gvFYstymDzaFN?e=o2g9km


Training of Decision Trees 
--------------------------

Each file of above dataset could be used to train a set of decision trees. For example, we use hv_16_16.npy to train the decision trees to decide to split horizontal or vertically. Consequently, we use have trained 21 random forests using sklearn library. We have took number of esimators, max depth and minimum samples of split as 40, 20 and 100 respectively.
The trained models are stored in pickle files. Since the size of random forest models depends on the number of training samples, we have specially picked numbers of training samples for each random forest to yield models with nearly the same size as original models provided by author.
The training of random forest models are done by running "rf_train_models_qm.py" and "rf_train_models_hv.py" in folder script/training. By executing these scripts, predictions of each decision tree in the random forest on a test set are also generated and saved.



Implementation of Decision Trees in VTM10 
--------------------------

The scripts involved in this section is in script/implementation. In the original paper, a specialised tree selection algorithm is presented. This algorithm aims at finding the best subset of decision trees in a random forest model so that the highest prediction accuracy is reached. By reusing the prediction results of decision trees obtained in the previous part, we execute the script eval_rf_models.py to evaluate the 
performance of different subsets of decision trees. In the end, the execution of get_tree_num.py is needed to show the best subset of decision trees for each random forest model. We use the sklearn-porter library to convert the trained random forest to C code as demonstrated in convert_rf.sh. Then we copie past the C code into rfTrainHor.cpp and rfTrainHor.cpp
as the definition of these classifiers. In the same time, the indices of decision trees of best subset are defined in file rfTrain.h. To evaluate the performance of our reproduction, you should build the project with macro COLLECT_DATASET desactivated. To regulate the level of acceleration, you should use the command line option -thdt which is a threshold for 
the prediction of decision trees.





**For reusing the code in this project, please think about citing paper [1] and [2]. Thanks!**


[1] G. Kulupana, V.P. Kumar M, and S. Blasi. Fast versatile video coding
using specialised decision trees. In 2021 Picture Coding Symposium
(PCS), pages 1–5, 2021

[2] .....

[3] D. Ma, F. Zhang, and D.R. Bull. Bvi-dvc: A training database for deep
video compression. IEEE Transactions on Multimedia, 24:3847–3858,
2021.

[4] Y. Wang, S. Inguva, and B. Adsumilli. Youtube ugc dataset for video
compression research. In 2019 IEEE 21st International Workshop on
Multimedia Signal Processing (MMSP), pages 1–5. IEEE, 2019





