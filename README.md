# :page_facing_up: Domain Specific Convolution and High Frequency Reconstruction based Unsupervised Domain Adaptation for Medical Image Segmentation (DoCR)

<p align="center"><img src="https://github.com/ShishuaiHu/DoCR/blob/master/figures/overview.png" width="90%"></p>

### Data Preparation

The dataset used in this study (RIGA+) can be downloaded from [Zenodo](https://zenodo.org/record/6325549).

You should download the dataset and unzip it.

### Dependency Preparation

```shell
cd DoCR
# Python Preparation
virtualenv .env --python=3
source .env/bin/activate
# Install PyTorch, compiling PyTorch on your own workstation is suggested but not needed.
# Follow the instructions on https://pytorch.org/get-started/locally/
pip install torch torchvision # or other command to match your CUDA version
# Install DoCR
pip install -e .
```

### Model Training and Inference

```shell
# Path Preparation
export OUTPUT_FOLDER="YOUR OUTPUT FOLDER"  # cannot be ended with '/'
export RIGAPLUS_DATASET_FOLDER="RIGA+ DATASET FOLDER"  # cannot be ended with '/'

# Target Domain - Base 1
DoCR_train --model DoCR --gpu 0 --tag Base1 \
--log_folder $OUTPUT_FOLDER \
--root $RIGAPLUS_DATASET_FOLDER \
--tr_csv $RIGAPLUS_DATASET_FOLDER/BinRushed_train.csv \
$RIGAPLUS_DATASET_FOLDER/BinRushed_test.csv \
$RIGAPLUS_DATASET_FOLDER/Magrabia_train.csv \
$RIGAPLUS_DATASET_FOLDER/Magrabia_test.csv \
--tu_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base1_unlabeled.csv \
--ts_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base1_test.csv 

# Target Domain - Base 2
DoCR_train --model DoCR --gpu 0 --tag Base2 \
--log_folder $OUTPUT_FOLDER \
--root $RIGAPLUS_DATASET_FOLDER \
--tr_csv $RIGAPLUS_DATASET_FOLDER/BinRushed_train.csv \
$RIGAPLUS_DATASET_FOLDER/BinRushed_test.csv \
$RIGAPLUS_DATASET_FOLDER/Magrabia_train.csv \
$RIGAPLUS_DATASET_FOLDER/Magrabia_test.csv \
--tu_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base2_unlabeled.csv \
--ts_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base2_test.csv 

# Target Domain - Base 3
DoCR_train --model DoCR --gpu 0 --tag Base3 \
--log_folder $OUTPUT_FOLDER \
--root $RIGAPLUS_DATASET_FOLDER \
--tr_csv $RIGAPLUS_DATASET_FOLDER/BinRushed_train.csv \
$RIGAPLUS_DATASET_FOLDER/BinRushed_test.csv \
$RIGAPLUS_DATASET_FOLDER/Magrabia_train.csv \
$RIGAPLUS_DATASET_FOLDER/Magrabia_test.csv \
--tu_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base3_unlabeled.csv \
--ts_csv $RIGAPLUS_DATASET_FOLDER/MESSIDOR_Base3_test.csv 
```

### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@inproceedings{hu2022domain,
  title={Domain Specific Convolution and High Frequency Reconstruction based Unsupervised Domain Adaptation for Medical Image Segmentation},
  author={Shishuai Hu and Zehui Liao and Yong Xia},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022},
  organization={Springer}
}
```

### Acknowledgements

- The code of Fourier Transform is adopted from [FDA](https://github.com/YanchaoYang/FDA). 
