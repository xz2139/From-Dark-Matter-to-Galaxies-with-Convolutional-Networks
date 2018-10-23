# darkness_beyond_time
dark matter simulation mapping to full cosmological similution 

Goal: estimate cosmological parameters (number of galaxies, dark matter mass, etc.) from dark matter simulation.

Data: 6 simulations from Illustris Project

Baseline Model: U-Net

Usage: python main.py
--mini  1 use only two cubes to train  (default 0)
--medium 1 use 4% of of the data to train  (default 0)
--lr  (default 0.01)
--epochs  (default 20)
--batch_size  (default 16)
--loss wight: weight of the loss equals to normalized [x, loss_weight * x,loss_weight * x] default 16
--model_idx 0: Unet 1:baseline  (default 0)
