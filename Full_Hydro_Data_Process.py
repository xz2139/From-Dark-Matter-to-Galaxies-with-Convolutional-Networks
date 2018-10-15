import h5py
from itertools import product
import pandas as pd
import numpy as np

import os
def process(postable):
    postable['x_b']=np.floor(postable['x']/(75000/1024))
    postable['y_b']=np.floor(postable['y']/(75000/1024))
    postable['z_b']=np.floor(postable['z']/(75000/1024))
    postable['x_b'] = postable['x_b'].astype(int)
    postable['y_b'] = postable['y_b'].astype(int)
    postable['z_b'] = postable['z_b'].astype(int)
    return postable

np.save('/scratch/xz2139/Full_zeros.npy',np.zeros((1024,1024,1024)))
for name in sorted(os.listdir('/scratch/xz2139/cosmo_full'))[1:]:
    filename = '/scratch/xz2139/cosmo_full/'+name
    f = h5py.File(filename, 'r')
    position=np.array(f['Subhalo']['SubhaloPos'])
    postable=pd.DataFrame(position)
    postable.columns=(['x','y','z'])
    mergetable=process(postable) 
    mergetable['c']=1
    def dataframe_to_array(df, out_shp):
        ids = np.ravel_multi_index(df[['x_b','y_b','z_b']].values.T, out_shp)
        val = df['count'].values
        return np.bincount(ids, val, minlength=np.prod(out_shp)).reshape(out_shp)

    counts=mergetable.groupby(['x_b','y_b','z_b'])['c'].count().reset_index(name="count")
    arr=dataframe_to_array(counts, (1024,1024,1024))
    old_arr=np.load('/scratch/xz2139/Full_zeros.npy')
    np.save('/scratch/xz2139/Full_zeros.npy',old_arr+arr)
    print(name)