# GazeVerification
Eye movement biometrics (EMB) research project

## Done Checklist:

- [x] Utilities and Data Transfer Objects (DTO) structures for data handling;
- [x] Release preprocessing scripts;
- [x] Construct Model structure for easy experiments setup: [Embedder + Body + Predictor Head];
- [x] Collect environment requirements for quick installation;
- [ ] Create run training & inference scripts;
- [ ] Demo notebooks with Colab hosting links;
- [ ] Release results!

## Introduction 
This repository contains useful features and tools for developing a new robust approach towards **gaze-based biometric person authentication** for the real-world applications usage.



## Install
### Setup `python` environment
```
conda create -n eyeverif python=3.7   # You can also use other environment.
```
### Install other dependencies
```
pip install -r requirements/requirements_basic.txt   # Main requirements list, other packages and libraries are optional for visualization or IO purposes
```

## Run Scripts

1) Select model type & parameters;
2) Save them to config files;
3) Run `run_train.py` or `run_inference.py`

## Contact

I. A. Abdullaeva (ireneabdullaeva.a@gmail.com)
