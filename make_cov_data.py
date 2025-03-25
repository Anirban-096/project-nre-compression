import os
import numpy as np
from tqdm import tqdm
from script import two_lpt
import script  
import matplotlib.pyplot as plt

def get_21cm_data_seed(seedvalue):

        ## Create an array of redshift values where snapshots will be created.

        data_dir = './simulation_files/files_seed'+str(seedvalue)

        savefile_noisecube= os.path.join(data_dir, 'noise21cm_cube_seed'+str(seedvalue)+'.npz')
        
        noisecube = np.load(savefile_noisecube)['noisecube_21cm']                       ## units : mK
        
        #datacube = Delta_Tb_arr + noisecube
        
        #cylindrical_pow_spec_binned_data, kount = ionization_map.get_binned_powspec_cylindrical(datacube, kpar_edges, kperp_edges, convolve='True', units='') 
        ## units are mK^2 h^-3 cMpc^3 
                
                        
        savefile_data = os.path.join(data_dir, 'data21cm_seed'+str(seedvalue)+'.npz')        
        # np.savez(savefile_data, datacube_21cm = datacube, cylindrical_pow_spec_binned_data = cylindrical_pow_spec_binned_data, kpar_bins = kpar_bins, kperp_bins = kperp_bins, kount = kount)   
        sfile=np.load(savefile_data)
        datacube_21cm = sfile['datacube_21cm']
        cylindrical_pow_spec_binned_data = sfile['cylindrical_pow_spec_binned_data']

        return cylindrical_pow_spec_binned_data.flatten()

def compute_correlation_matrix(cov_matrix):
    # Extract standard deviations from the covariance matrix
    sigma = np.sqrt(np.diag(cov_matrix))

    # Compute outer product of standard deviations for normalization
    sigma_outer = np.outer(sigma, sigma)

    # Element-wise division to compute correlation matrix
    corr_matrix = cov_matrix / sigma_outer

    # Ensure the diagonal is exactly 1
    #np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix

############ MAIN #######################
seedlist=np.genfromtxt('mockdata_seed_list.txt',dtype=int) ## the first 50 seeds of musicseed_list.txt

datamatrix=[]
#datamatrix = np.zeros(shape=(len(seedlist), 20**2))

for i, seed in enumerate(tqdm(seedlist, desc="Fetching mock 21cm data")):
    pspec = get_21cm_data_seed(seed)
    #print(pspec)
    datamatrix.append(pspec)

datamatrix=np.array(datamatrix)

cov_matrix = np.cov(datamatrix, rowvar=False, bias=False)

corr_matrix = compute_correlation_matrix(cov_matrix)

plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, cmap='viridis', aspect='auto',vmin=-1,vmax=1)
plt.colorbar(label='')
plt.title("Correlation Matrix of $\Sigma_{data}$")
#plt.xlabel("h/cMpc")
#plt.ylabel("h/cMpc")
plt.show()
