# Stacking Machine Learning Models for Predicting Photophysical Properties of Iridium Complexes

## 1 Browse all models
Details of all trained models are gathered in several directories, which has a prefix **bm**. The following list shows the meaning of the suffix.
+ em4: models predicting 4 emission tasks: emlb, sigma, ltl10, integ
+ mt: multi-task models which predicting the max absorption wavelength of complexes in the absorption dataset.
+ qf: models which feature generation without electronic structure calculation


In each directory, the name of csv file includes the predicting target, the learner layer, and filtering label. The following list shows the meaning of the suffix.

+ cf: the best learner of each complex feature set 
+ cffs: the best learner of each complex feature set and feature selection method
+ cffsl: all learners of the target

Take the task emission wavelength for example, after the process of feature preprocessing and selection, totally 3854 complex feature subsets are generated from 286 remaining complex feature sets. Apart from the same feature subsets, there are 3612 feature subsets actually for base learner training. 

Here, in the directory **bm_em4**, the file **emlb_base-learner_cf.csv** lists 286 base learners which perform best among base learners of the same complex feature set. The file **emlb_base-learner_cffs.csv** list 3854 base learners which are the best among base learners of the same feature subset, as mentioned above.  The file **emlb_base-learner_cffsl.csv** contains all base learners predicting emission wavelength of complexes.

## 2 Using optimized parameter for model prediction
Directory **op_pred** contain an example to show how we use trained model for predicting new complex data, see **op_prediction.ipynb**. 

## 3 Training stacking models from zero
The directory **mfo** contains necessary data and code for building stacking models from zero.
Note that the directory are located in `https://figshare.com/articles/dataset/irml_mfo/27683360` naming **irml_mfo**
Note that all the files and results based on QM-calculated features are located in **irml_mfo** naming **xqm2**
### 3.1 Gathering ligand structures and targets of each metal complex
The whole project begins with gathering data about the relationship between ligand structures, combination of complexes, and their photophysical properties. Note that feature names here are different from the paper.

+ For emission task based on the emission data set, as is shown in **op_pred**, we put forward a demo for generating '13cn' complex feature set. Other mentioned complex feature sets are listed in directory **irja_complex_x**. Meanwhile, emission targets are listed in **irja_complex_y**

+ For multi-task models, which predict the max absorption wavelength of 100 complexes, complex data are listed in **ob100_complex_ori**, along with predicting target which are listed in **ob100_complex_y**

### 3.2 Data splitting and preprocessing
For training stacking models predicting emission tasks, run **ij_fsf_slf.py** for calling corresponding object, and generating varies of data set in directory **result__\***.
### 3.3 Feature selection and data checking
The first stage of feature selection are developed in the previous section, that is 'preprocessing' Then, in this work, we provide three types of feature selection methods: random selection, filtering selection and forward selection. Files **ij_fsijs_slf.py** and **ij_fsijs_mm_ori.py** lists all mentioned methods.

In this procedure, a new directory  **snresult_\*** is generated, for the storage of feature selection logs and data for base learner layer according to different predicting targets.

Then, gather all logs about feature selection methods, and fill into **slf_ijsb.py** for checking. If no conflict report here, file **job_list_typical_.txt** will be generated in directory **snresult_\***, this is an important schedule for base learner training.

### 3.4 Base learner training
File **irja_ijsc_jobs__0.py** shows an example about base learner training, which output log files in the directory **snresult_\***.

After finish all base learner training, a extra check is arranged for check model data and store predicting results. Here, file **slf_ijscre_jobs__163.py** is an example.

### 3.5 Meta learner training
**slf_ijsd_jobs__0.py** is an example for meta learner training.

## 4 Multi-task meta learner training
### 4.1 Predicting absorption data using base learners from 'emlb' task
File **slf_emlb_ijsta_jobs__0.py** is an example to show how transform complex data and model parameters from an old data set to a new one, so it seems more complicated than directly arranging a base learner training.
### 4.2 Training meta learner for predicting absorption data
File **slf_emlb_ijstb_jobs__0.py** is an example for training multi-task meta learners. Comparing with typical meta learner training, logs and predicting results of these multi-task meta learner are stored in new directory. Besides, we also tried several random seeds for splitting the absorption data set.

## 5 Machine learning without electronic structure calculation
### 5.1 Preparing data for training classification models
Complex feature sets are stored in directory **irja_complex_xoa**, predicting targets about sorting are stored in **irja_complex_yoa**.
### 5.2 Data splitting and preprocessing
For training classification models, here **oa_fsf.py** provides detailed configuration for data splitting, the result are stored in **result__oa2**

Then, like **ij_fsijs_slf.py** and **ij_fsijs_mm_ori.py** mentioned above, there's also a feature selection procedure training classification models. Also, there's a **ijsb** procedure for checking generated data. Finally, directory **snresult_oa2** is generated for training classification models.

### 5.3 Training classification models and generating new complex feature sets
The file **oa_ijsclarere__5.py** is an example for training classification models. After finishing all models, use **oa2n_oplog.py** to browse predicting ability of all classification models, and generating new complex feature sets into directory **irja_complex_xoa2n** and **snresult_slfoa2n**

### 5.4 Using QM-free complex feature sets training base learners
To easily connect and transform QM-free feature sets and original generated complex feature sets, we use several files which are similar with multi-task model. Here, files **slf_emlb_oa2nt_ijsta.py**,**slf_emlb_oa2no_ijsta.py**, and **slf_emlb_oa2nr_ijsta.py** arranges model training to predict training sets, duality sets and test sets. In this way, we gather enough predicting result for training meta learners, located in directories **snresult_slfoa2nt**, **snresult_slfoa2no** and **snresult_slfoa2nr**.

### 5.5 Training QM-free meta learners
Likewise, file **slf_emlb_oa2n_tro_ijsd3re_gen.py** generates configure files for meta learner training.



## 6 copyright and citation
+ This work is registered as a software with the code **2024SR0831144**
+ The paper is published: *J. Photoch. Photobio. A*, **2025**, 466, 116374. DOI: 10.1016/j.jphotochem.2025.116374