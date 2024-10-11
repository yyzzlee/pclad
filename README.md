# The source code for the paper "Deep Anomaly Detection with Partition Contrastive Learning for Tabular Data" submitted to the ECML-PKDD 2025 journal track.

## Environment  
main packages
```  
torch==1.12.1+cu114  
numpy==1.23.5  
pandas==1.1.3  
scipy==1.5.2  
scikit-learn==0.22  
```  
we provide a `requirements.txt` in our repository.


  
## Datasets used in our paper  

The used datasets can be downloaded from:  
- Thyroid     https://odds.cs.stonybrook.edu/thyroid-disease-dataset/
- Arrhythmia  https://odds.cs.stonybrook.edu/arrhythmia-dataset/
- Bank        https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/tree/main/numerical%20data/DevNet%20datasets
- Celeba      https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/tree/main/numerical%20data/DevNet%20datasets
- Secom       https://www.kaggle.com/datasets/paresh2047/uci-semcom
- UNSW_NB15   https://research.unsw.edu.au/projects/unsw-nb15-dataset 
- CIC-IDS2017 https://www.unb.ca/cic/datasets/ids-2017.html
- ADBench     https://github.com/Minqi824/ADBench/
  
  
## Reproduction of experiment results
### Anomaly Detection Performance
use `script_effectiveness.sh` produce anomaly detection results.
```shell
./script_effectiveness.sh
``` 

### Robustness w.r.t. Different Contamination Levels
use `script_robustness.sh` produce detection results w.r.t. different contamination rates in the training data.
```shell
./script_robustness.sh
```

### Ablation study
use `script_ablation.sh` produce detection results of ablated variants (w/o L_LCL, w/o L_GSC and w/o Part).
```shell
./script_ablation.sh
```

### Parameter Sensitivity 
use `script_sensitivity_.sh` produce detection results with different hyperparameters.
```shell
./script_sensitivity_d.sh    #hyperparameter d
``` 
```shell
./script_sensitivity_n.sh    #hyperparameter n
``` 
```shell
./script_sensitivity_lamda.sh    #hyperparameter lamda
``` 
```shell
./script_sensitivity_lr.sh    #hyperparameter lr
``` 
```shell
./script_sensitivity_m.sh    #hyperparameter m
``` 
Please note that when testing the sensitivity of the hyperparameter m (the number of partitions), it is necessary to first remove the update dictionary for it from the config.py.


## Competing methods
All the anomaly detectors in our paper are implemented in Python. We list their publicly available implementations below. 
- `iForest`: we use the [scikit-learn] package
- `ICL`: https://openreview.net/forum?id=_hszZbt46bT
- `NeuTraL`: https://github.com/boschresearch/NeuTraL-AD
- `GOAD`: https://github.com/lironber/GOAD 
- `RCA`: https://github.com/illidanlab/RCA
- `GAAL`, `LOF`, `OCSVM`, `COPOD` and `ECOD`: we directly use [pyod](https://github.com/yzhao062/Pyod) (python library of anomaly detection approaches)
- `DSVDD`: https://github.com/lukasruff/Deep-SVDD-PyTorch 
- `SLAD` and `DIF`: https://github.com/xuhongzuo/DeepOD
- `SLA2P`: https://github.com/wyzjack/SLA2P
