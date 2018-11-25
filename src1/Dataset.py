from torch.utils import data
import numpy as np
# cat means what target data we want to use, 
#'count' means the count of the galaxy, and 'mass' means the mass of the galaxies.
#when convert = 1 we have the classification problem, otherwise we have the regression problem
class Dataset(data.Dataset):
    def __init__(self, lists, cat = 'count', aug = False, reg = True):
        'Initialization'
        self.IDs = lists
        if cat != 'count' and cat != 'mass':
            raise ValueError('cat not exist')
        self.cat = cat
        self.aug = aug
        self.reg = reg
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.IDs)

    def convert_class(self, num):
        if num==0:
            return 0
        elif num>0:
            return 1
        else:
            print('dark matter mass smaller than 0')



    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.IDs[index]
        d_box = np.load('/scratch/xz2139/cosmo_dark/arrays/'+str(ID[0])+'_'+str(ID[1])+'_'+str(ID[2])+'.npy')
        #dm_mean = 5.614541471004486
        #d_box = (d_box - dm_mean) / dm_mean
        if self.cat == 'count':
            f_box=np.load('/scratch/xz2139/cosmo_full/arrays/'+str(ID[0])+'_'+str(ID[1])+'_'+str(ID[2])+'.npy')
            if not self.reg:
                convert= np.vectorize(self.convert_class) #Convert python function to vector function
                f_box=convert(f_box)
        elif self.cat == 'mass':
            f_box=np.load('/scratch/xz2139/cosmo_mass/arrays/'+str(ID[0])+'_'+str(ID[1])+'_'+str(ID[2])+'.npy')
        
        if self.aug:
            dim_to_flip = tuple(np.arange(3)[np.random.choice(a= [False, True], size = 3)])
            if len(dim_to_flip) > 0:
                d_box = np.flip(d_box,dim_to_flip)
                f_box = np.flip(f_box,dim_to_flip)

        return d_box,f_box



