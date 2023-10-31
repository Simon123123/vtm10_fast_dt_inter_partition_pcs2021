Fast Decsion Tree based Inter Partitioning of VVC (PCS 2021)
============================================================

This is the reproduction of paper **Fast Versatile Video Coding using Specialised Decsion Trees** [1] by Gosala Kulupana et al. published in Picture Coding Symposium 2021. Many thanks to Mr kulupana for sharing the source code of his implementation in VTM8 with me. This serves as the stat of art comparaison in our paper **CNN-based Prediction of Partition Path for VVC Fast Inter Partitioning Using Motion Fields** [2] currently under review of IEEE Transaction of Image Processing.


Comparing to original paper, we have generated a new dataset and trained the decision tree classifers. Then the proposed method and trained decision trees are integrated into VTM10. With the approvement of Kulupana, we share the dataset, trained decision tree, and all the related code in this repository. Python scripts for processing the raw data and training && pruning the decision tree are included in folder scripts. The **master branch** is our reproduction of the paper while the branch 
**original_implementation** contains the souce code in VTM8 offered by the author.   



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

In the original paper, the dataset was constructed using a portion of the Common Test Condition (CTC) sequences, while the remaining sequences were used for evaluating the method. In our implementation, we chose to generate the dataset using the BVI-DVC database [3] and the Youtube UVG database [4] to assess performance on the full CTC. All sequences in these databases have been encoded. Data is selectively collected from certain inter frames of the encoded sequences. More precisely, we collect data from one frame every three frames for resolutions of 960x544 and 480x272. For other resolutions, we specifically collect data from frames with a POC (Picture Order Count) equal to 8, 16, 28, 42, 49. 

To adjust the frames for data collection, please modify the **frame_collect** variable, which is defined in line 1551 and 2905 in the file **EncModeCtrl.cpp**. The collected features are stored in two generated csv files, namely **split_cost_yuvname.csv** and **split_features_yuvname.csv**. To generate the dataset, please run the encoding with the macro **COLLECT_DATASET**
activated, which can be found in line 68 of the file **TypeDef.h**. After obtaining the csv files, you will need several Python scripts to process the collected data.





1. Firstly, the script **sep_frames.py** is executed to seperate the collected data by frames. For 4k resolution, the frame-level data is then split by rows after running the script **sep_rows.py**. Splitting the large data file facilitates the multiprocessing of files.

2. Then we execute the script **feature_ext.py** to extract the features for QT/MT decisions and for Horzontal/Vertical decisions, respectively, for each data file.

3. Finally the feature files are merged per CU size and per decision. An equal number of samples for each decision has been randomly selected. In this stage, we begin by running **sep_cu_sizes.py** followed by the execution of the **merge_features.py** script.


The scripts for the above steps are provided in the **scripts/processing** folder. Finally, a dataset containing 21 numpy files is generated. If you are looking for details about the data format in this dataset, please refer to the code and the original paper. 


Here is our dataset: https://1drv.ms/f/s!Aoi4nbmFu71Hgn0gvFYstymDzaFN?e=o2g9km


Training of Random Forests 
--------------------------

Each file in the dataset mentioned above can be used to train a random forest model. For example, we use **hv_16_16.npy** to train the random forest for deciding between horizontal and vertical splits. Consequently, we trained 21 random forests using the scikit-learn library. We use 40 esimators, a maximum depth of 20, and a minimum number of samples per split of 100 for training these models.
The trained models are stored in pickle files. Since the size of random forest models depends on the number of training samples, we select a specific number of training samples for each random forest to produce models with nearly the same size as the original models provided by the author.
The training of random forest models is performed by running **rf_train_models_qm.py** and **rf_train_models_hv.py** in the **scripts/training** folder. When these scripts are executed, predictions of each decision tree in the random forest on a test set are also generated and saved.




Integration of Decision Trees in VTM10 
--------------------------

The scripts involved in this section are located in the **scripts/integratio** directory. In the original paper, a specialised tree selection algorithm is presented. This algorithm aims to find the best subset of decision trees in a random forest model to achieve the highest prediction accuracy. By reusing the prediction results of decision trees obtained in the previous part, we execute the script **eval_rf_models.py** to evaluate the 
performance of different subsets of decision trees. Finally, the execution of **get_tree_num.py** is necessary to determine the best subset of decision trees for each random forest model.



We use the **sklearn-porter** library to convert the trained random forest into C code, as demonstrated in **convert_rf.sh**. Then we copy and paste the C code into **rfTrainHor.cpp** and **rfTrainHor.cpp**
as the definitions of these classifiers. At the same time, the indices of decision trees of the best subset are defined in the file **rfTrain.h**. To evaluate the performance of our reproduction, you should build the project with the macro **COLLECT_DATASET** deactivated. To adjust the level of acceleration, you should use the command-line option **-thdt** which sets a threshold for the prediction of decision trees.



**For reusing the code in this project, please think about citing paper [1] and [2]. Thanks!**
**If you have further questions, please contact me at liusimon0914@gmail.com.**


[1] G. Kulupana, V.P. Kumar M, and S. Blasi. Fast versatile video coding
using specialised decision trees. In 2021 Picture Coding Symposium
(PCS), pages 1–5, 2021

[2]  Y. Liu, M. Riviere, T. Guionnet, A. Roumy, and C.Guillemot. CNN-based Prediction of Partition Path for VVC Fast Inter Partitioning Using Motion Fields. ArXiv, abs/2310.13838. 
(cf. https://arxiv.org/abs/2310.13838) 

[3] D. Ma, F. Zhang, and D.R. Bull. Bvi-dvc: A training database for deep
video compression. IEEE Transactions on Multimedia, 24:3847–3858,
2021.

[4] Y. Wang, S. Inguva, and B. Adsumilli. Youtube ugc dataset for video
compression research. In 2019 IEEE 21st International Workshop on
Multimedia Signal Processing (MMSP), pages 1–5. IEEE, 2019





