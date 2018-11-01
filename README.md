# Darkness Beyond Time 

### Topic: dark matter simulation mapping to full cosmological similution 

### Goal: estimate cosmological parameters (number of galaxies, dark matter mass, etc.) from dark matter simulation.

### Data: 6 simulations from Illustris Project

### Baseline Model: U-Net


### Usage: python main.py
### --mini  1 use only two cubes to train  (default 0)
### --medium 1 use 4% of of the data to train  (default 0)
### --lr  (default 0.01)
### --epochs  (default 20)
### --batch_size  (default 16)
### --loss wight: weight of the loss equals to normalized [x, loss_weight * x,loss_weight * x] default 16
### --model_idx 0: Unet 1:baseline  (default 0)

## Memo on 10/30:

To generate data for training, the whole simulation was chopped into 32^3=32768 boxes. Each Box is splitted into 32^3=32768 sub-boxes. We count the number of data matters and subhalos within each sub-boxes. And store each box as array of size (32, 32, 32). We discovered that our target data are exceptionally sparse as zeros in targets are about 99.5% among all sub-boxes. Hence we first convert it to a classification problem, reclassified the subhalo array to have 0 when there is no galaxy in the sub-box and 1 when there is 1 or more galaxies. The reclassifies array is still of dimension (32, 32, 32) but with only zeros and ones. 

We first finished our baseline Model with 5-Layer U-Net. Trained on 60% of all dataset for 240 epoches. We also played around with the weight of cross entropy loss to counterpart the sparsity. The model reached accuracy of 99.29 percent while having a recall of 62 percent and percision of 29 percent. We trained another model with one-layer convolution of kernel size (3,3,3), following by nonelineaity and a fully connected layer. The simple model works pretty well in detecting the existence of galaxy, with model accuracy 97%, recall 98%, and false positive rate 4%. We also tried R2-Unet(Residule Recurrent Unet) which is a deeper version of unet. After running for 50 epochs with 25% data, the model achieved accuracy of 97%, recall of 81%. 
