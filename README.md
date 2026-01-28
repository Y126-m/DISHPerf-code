
# DISHPerf: Deep Interaction-aware SHallow Network for Performance Prediction of Configurable Systems
This drive releases the code and data for the DISHPerf model, an improved shallow neural network architecture with explicit interaction-aware modeling for configurable system performance prediction.
## Dependencies

+ Python 3.8.5
+ Numpy 1.20.3
+ Tensorflow 1.15.0
+ Seaborn 0.11.2
+ Matplotlib 3.5.1

## Directories

+ `datasets`: dataset including the performance data of the ten subject systems.
+ `models`: python code for building the DISHPerf model.
+ `utils`: python code for auxiliary training.

## Usage

To run DISHPerf, users need to specify the name of the software system to evaluate and execute the script `run.py`. The model supports 10 configurable systems: x264, BDBJ, lrzip, polly, vp9, Dune, hipacc, hsmgp, javagc, sac. 
```shell
$ python run.py x264
```

When finishing each sample size, the script will output a .csv file that shows the mean prediction error and the margin (95% confidence interval) of that sample size over the 30 experiments. These results will be similar as the results we report in Table 2 of our paper.

