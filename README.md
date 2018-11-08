# Darkness Beyond Time 

### Topic: dark matter simulation mapping to full cosmological similution 

### Goal: estimate cosmological parameters (number of galaxies, dark matter mass, etc.) from dark matter simulation.

### Data: 6 simulations from Illustris Project

### The best result will be saved to all_results(along with the hyperparameters)

### Baseline Model: U-Net

### Sample script for running classification
python src/main.py --medium1 1 --lr 0.002 --loss_weight 20 --model_idx 0 --epochs 3 --target_cat 'count' --target_class 0 > result.txt
### Sample script for running regression
python src/main.py --medium 1 --lr 0.002 --loss_weight 20 --model_idx 0 --epochs 3 --target_cat 'mass' --target_class 1 > result_reg.txt



### --mini  1 use only two cubes to train  (default 0)
### --medium 1 use 10% of of the data to train  (default 0)
### --medium1 1 use 2% of of the data to train  (default 0)
### --print_freq 
### --lr  (default 0.01)
### --model_idx 0:Unet  1:baseline  2: Inception   (default 0)
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

Update on 10/30:

To generate data for training, the whole simulation was chopped into 32^3=32768 boxes. Each Box is splitted into 32^3=32768 sub-boxes. We count the number of data matters and subhalos within each sub-boxes. And store each box as array of size (32, 32, 32). We discovered that our target data are exceptionally sparse as zeros in targets are about 99.5% among all sub-boxes. Hence we first convert it to a classification problem, reclassified the subhalo array to have 0 when there is no galaxy in the sub-box and 1 when there is 1 or more galaxies. The reclassifies array is still of dimension (32, 32, 32) but with only zeros and ones. 

We first finished our baseline Model with 5-Layer U-Net. Trained on 60% of all dataset for 240 epoches. We also played around with the weight of cross entropy loss to counterpart the sparsity. The model reached accuracy of 99.29 percent while having a recall of 62 percent and percision of 29 percent. We trained another model with one-layer convolution of kernel size (3,3,3), following by nonelineaity and a fully connected layer. The simple model works pretty well in detecting the existence of galaxy, with model accuracy 97%, recall 98%, and false positive rate 4%. We also tried R2-Unet(Residule Recurrent Unet) which is a deeper version of unet. After running for 50 epochs with 25% data, the model achieved accuracy of 97%, recall of 81%. 
