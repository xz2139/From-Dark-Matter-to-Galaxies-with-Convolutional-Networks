# Darkness Beyond Time 

### Topic: dark matter simulation mapping to full cosmological similution 

### Goal: estimate cosmological parameters (number of galaxies, dark matter mass, etc.) from dark matter simulation.

### Data: 6 simulations from Illustris Project

### Code Usage see notebook **Using and developing the code**

### Sample script for running classification see **run_c.s**
### Sample script for running two-phase model see **run_all.s**

i



<!-- ### --mini  1 use only two cubes to train  (default 0)
### --medium 1 use 10% of of the data to train  (default 0)
### --medium1 1 use 2% of of the data to train  (default 0)
### --print_freq 
### --lr  (default 0.01)
### --model_idx 0:Unet, 1:baseline 2: Inception 3. R2Unet 4.two-phase model(classfication phase: one layer Conv, regression phase: R2Unet) 5.two-phase model(classfication phase: R2Unet, regression phase: R2Unet)   
### --epochs  (default 20)
### --batch_size  (default 16)
### --loss wight: weight of the loss equals to normalized [x, loss_weight * x]
(default 20)
### --target_cat: the feature we want to predict. count: the count of the subhalos mass: the mass of the subhalos   (default 'count')
### --target_class：  classification problem(0) or regression problem(1)   (default 0)
### --plot_label:  label for the filename of the plot. If left default, the plot_label will be '_' + target_class + '_' + target_cat. This label is for eliminating risk of overwriting previous plot 
### --load_model 
### --record_results whether to write the best results to all_results.txt

### Find the plot of trainning loss, validation loss, validation accuracy, validation recall and validation precision in the ./fig folder
 -->
