import h5py
from itertools import product
import pandas as pd
import numpy as np

import os
e=0.0001

output_grid='/scratch/xz2139/Starm_zeros.npy'
raw_data='/scratch/xz2139/cosmo_Starm'
out_data='/scratch/xz2139/cosmo_Starm/arrays'
def process(postable):
    postable['x_b']=np.floor(postable['x']/((75000+e)/1024))
    postable['y_b']=np.floor(postable['y']/((75000+e)/1024))
    postable['z_b']=np.floor(postable['z']/((75000+e)/1024))
    postable['x_b'] = postable['x_b'].astype(int)
    postable['y_b'] = postable['y_b'].astype(int)
    postable['z_b'] = postable['z_b'].astype(int)
    return postable

np.save(output_grid,np.zeros((1024,1024,1024)))
for name in sorted(os.listdir(raw_data)):
    filename = rawdata+'/'+name
    f = h5py.File(filename, 'r')
    position=np.array(f['Subhalo']['SubhaloPos'])
    if len(position)==0:
        pass
    postable=pd.DataFrame(position)
    mass=np.array(f['Subhalo']['SubhaloMassType'][:,5])
    masstable=pd.DataFrame(mass)
    masstable.columns=(['m'])
    postable.columns=(['x','y','z'])
    mergetable=process(postable) 
    mergetable['c']=masstable['m']
    def dataframe_to_array(df, out_shp):
        ids = np.ravel_multi_index(df[['x_b','y_b','z_b']].values.T, out_shp)
        val = df['sum'].values
        return np.bincount(ids, val, minlength=np.prod(out_shp)).reshape(out_shp)
    counts=mergetable.groupby(['x_b','y_b','z_b'])['c'].sum().reset_index(name="sum")
    arr=dataframe_to_array(counts, (1024,1024,1024))
    old_arr=np.load(output_grid)
    np.save(output_grid,old_arr+arr)
    print('Finished: '+ name)



pos=list(np.arange(0,1024,32))
ranges=list(product(pos,repeat=3))
print('Generating subboxes....')
for ID in ranges:
    f_box=output_grid[ID[0]:ID[0]+32,ID[1]:ID[1]+32,ID[2]:ID[2]+32]
    np.save(out_data+'/'+str(ID[0])+'_'+str(ID[1])+'_'+str(ID[2])+'.npy',f_box)
print('All finished')