###
# This example shows how to run MUSIC to generate 2LPT particle positions
# and velocities that are needed for ionization maps.
###

import os
import numpy as np
from tqdm import tqdm
from script import two_lpt
import script
import tools21cm as t2c

def get_21cm_noise_seed(seedvalue):

        ## Create an array of redshift values where snapshots will be created.

        data_dir = './simulation_files/files_seed'+str(seedvalue)

        ## script parameters

        ngrid=128
        box = 256.0 ### Mpc/h
        h = 0.678
        
        z = 7.0

        box_cMpc = box/h

        ## radio observation parameters

        total_int_time = 6.0 #hours, per day observations
        int_time  = 10       #seconds, intergration time
        depth_mhz = 8       #MHz, bandwidth
        obs_time  = 1080     #hours, total observation time
      
        noisecube = t2c.noise_cube_coeval(seedvalue,
                            ngrid, 
                            z, 
                            depth_mhz= depth_mhz ,
                            filename = None, 
                            obs_time = obs_time,
                            total_int_time = total_int_time,
                            int_time = int_time,
                            boxsize  = box_cMpc, 
                            uv_map   = None,
                           )
        ## units of mK

        savefile_noisecube= os.path.join(data_dir, 'noise21cm_cube_seed'+str(seedvalue))

        np.savez(savefile_noisecube, noisecube_21cm = noisecube)



############ MAIN #######################
seedlist=np.genfromtxt('mockdata_seed_list.txt',dtype=int) ## the first 50 seeds of musicseed_list.txt

for i, seed in enumerate(tqdm(seedlist, desc="Running tools21cm")):
    get_21cm_noise_seed(seed)

