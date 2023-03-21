# Cross Validation Experiments 

## About
In the paper we carried a 5-fold cross validation using PEGASUS-X and LongT5.

To reproduce the experiments, one can just run the script which creates a new Python environmnet for the experiment.

The 5-fold Cross Validation dataset are located in this folder for easiness.


## Run the experiments

Make sure that have CUDA installed in your computer. 
A version of PyTorch will be installed, you may need to install a different version depending on your CUDA configuration.

The hyper-params are for A-100 80Gb Ampere NVIDIA GPU card. Anything with less memory will struggle.

Run `install_experim_script.sh` from your terminal, from thed current directory.