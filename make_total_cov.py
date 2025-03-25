import os
import numpy as np
from tqdm import tqdm
from script import two_lpt
import script  
import matplotlib.pyplot as plt


def get_21cm_data_seed(seedvalue):

        ## Create an array of redshift values where snapshots will be created.

        data_dir = './simulation_files/files_seed'+str(seedvalue)
        
        #cylindrical_pow_spec_binned_data, kount = ionization_map.get_binned_powspec_cylindrical(datacube, kpar_edges, kperp_edges, convolve='True', units='') 
        ## units are mK^2 h^-3 cMpc^3 
                                 
        savefile_data = os.path.join(data_dir, 'data21cm_seed'+str(seedvalue)+'.npz')        
        # np.savez(savefile_data, datacube_21cm = datacube, cylindrical_pow_spec_binned_data = cylindrical_pow_spec_binned_data, kpar_bins = kpar_bins, kperp_bins = kperp_bins, kount = kount)   
        sfile=np.load(savefile_data)
        #datacube_21cm = sfile['datacube_21cm']
        cylindrical_pow_spec_binned_data = sfile['cylindrical_pow_spec_binned_data']

        return cylindrical_pow_spec_binned_data, cylindrical_pow_spec_binned_data.flatten()

    
def get_21cm_sim_seed(seedvalue):

        ## Create an array of redshift values where snapshots will be created.

        data_dir = './simulation_files/files_seed'+str(seedvalue)
        
        #cylindrical_pow_spec_binned_data, kount = ionization_map.get_binned_powspec_cylindrical(datacube, kpar_edges, kperp_edges, convolve='True', units='') 
        ## units are mK^2 h^-3 cMpc^3 
                
                        
        savefile_data = os.path.join(data_dir, 'sim21cm_seed'+str(seedvalue)+'.npz')        
        # np.savez(savefile_data, datacube_21cm = datacube, cylindrical_pow_spec_binned_data = cylindrical_pow_spec_binned_data, kpar_bins = kpar_bins, kperp_bins = kperp_bins, kount = kount)   
        #print(savefile_data)
        sfile=np.load(savefile_data)
        cylindrical_pow_spec_binned_data = sfile['cylindrical_pow_spec_binned_signal']
        #print(cylindrical_pow_spec_binned_data.shape, cylindrical_pow_spec_binned_data.flatten().shape)
        return cylindrical_pow_spec_binned_data, cylindrical_pow_spec_binned_data.flatten()
        
        
############ SIGMA_SIM #######################

        
sim_seedlist=np.genfromtxt('musicseed_list.txt',dtype=int)[50:] ## the last 50 seeds of musicseed_list.txt

simmatrix=[]
simmatrix_flattened=[]

for i, seed in enumerate(tqdm(sim_seedlist, desc="Fetching simulated 21cm data")):
    pspec_arr,  pspec_flattened = get_21cm_sim_seed(seed)
    simmatrix.append(pspec_arr)
    simmatrix_flattened.append(pspec_flattened)

simmatrix_flattened = np.array(simmatrix_flattened)

sim_cov_matrix_flattened = np.cov(simmatrix_flattened, rowvar=False, bias=False)


############ SIGMA_DATA #######################

data_seedlist=np.genfromtxt('mockdata_seed_list.txt',dtype=int) ## the first 50 seeds of musicseed_list.txt

datamatrix=[]
datamatrix_flattened=[]



for i, seed in enumerate(tqdm(data_seedlist, desc="Fetching mock 21cm data")):
    pspec_arr,  pspec_flattened  = get_21cm_data_seed(seed)
    datamatrix.append(pspec_arr)
    datamatrix_flattened.append(pspec_flattened)
    
datamatrix_flattened =np.array(datamatrix_flattened)

data_cov_matrix_flattened = np.cov(datamatrix_flattened, rowvar=False, bias=False)


############ SIGMA_TOTAL #######################

total_cov_matrix_flattened =  data_cov_matrix_flattened + sim_cov_matrix_flattened

np.savez('total_cov_50real_flattened.npz', total_cov_flat = total_cov_matrix_flattened)
