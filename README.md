# Unpaired Cross-modal Interaction Learning for COVID-19 Segmentation on Limited CT images [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43898-1_58)]

### Data preparation
Download [COVID-19-20 dataset](https://covid-segmentation.grand-challenge.org).

Preprocess the COVID-19-20 dataset according to the [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

Download [ChestXray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345).

Download [ChestXR dataset](https://cxr-covid19.grand-challenge.org).

Weight initialization：we follow the extension of [UniMiSS](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_33) for weight initialization. (coming soon)



File directory tree:
```
├─nnUNet
│  ├─nnUNet_preprocessed
│  │  │  Task115_COVIDSegChallenge # COVID-19-20 dataset
├─Chestxray # ChestXray14 dataset
│  ├─image
│  │  │  xxx.png
│  │  │  ...
├─CXR_Covid-19_Challenge # ChestXR dataset
│  ├─train
│  │  ├─covid
│  │  │  │  cov_xxx.jpg
│  │  │  │  ...
│  │  ├─normal
│  │  │  │  normal-xxx.jpg
│  │  │  │  ...
│  │  ├─pneumonia
│  │  │  │  pneumoniaxxx.jpg
│  │  │  │  ...
│  ├─validation
│  │  ├─covid
│  │  │  │  cov_xxx.png
│  │  │  │  ...
│  │  ├─normal
│  │  │  │  normal_xxx.png
│  │  │  │  ...
│  │  ├─pneumonia
│  │  │  │  pneu_xxx.png
│  │  │  │  ...
├─pretrain_ED # Weight initialization
├─UCI
│  │ *.py
│  │ ...
```

### Train with Chestxray14
```
python train_withChestXray14.py
```

### Train with ChestXR
```
python train_withChestXR.py
```

### Evaluate
```
python evaluate.py
```