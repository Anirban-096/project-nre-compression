import os
import numpy as np
from tqdm import tqdm
from script import two_lpt
import script  
import matplotlib.pyplot as plt

def get_k_edges(boxlength, numgrid, log_bins=False):

    nbins=0 # auto-calculate from boxsize and numgrids

    kmin = 2 * np.pi / boxlength
    kmax = np.pi * numgrid / boxlength

    if (kmax <= kmin):
        sys.exit('set_k_bins: kmax is less than kmin')

    if (log_bins):
        if (nbins < 2):
            dlnk = 0.2
            nbins = int((np.log(kmax) - np.log(kmin)) / dlnk)
        else:
            dlnk = (np.log(kmax) - np.log(kmin)) / (nbins)
        lnk_edges = np.linspace(np.log(kmin), np.log(
            kmax), num=nbins+1, endpoint=True)
        lnk_bins = (lnk_edges[:-1] + lnk_edges[1:]) / 2
        k_edges = np.exp(lnk_edges)
        k_bins = np.exp(lnk_bins)
    else:
        if (nbins < 2):
            dk = 0.1
            nbins = int((kmax - kmin) / dk)
        else:
            dk = (kmax - kmin) / (nbins)
        k_edges = np.linspace(kmin, kmax, num=nbins+1, endpoint=True)
        k_bins = (k_edges[:-1] + k_edges[1:]) / 2

    return k_edges, k_bins

def get_21cm_data_seed(seedvalue):

        ## Create an array of redshift values where snapshots will be created.

        data_dir = './simulation_files/files_seed'+str(seedvalue)

        #savefile_noisecube= os.path.join(data_dir, 'noise21cm_cube_seed'+str(seedvalue)+'.npz')
        
        #noisecube = np.load(savefile_noisecube)['noisecube_21cm']                       ## units : mK
        
        #datacube = Delta_Tb_arr + noisecube
        
        #cylindrical_pow_spec_binned_data, kount = ionization_map.get_binned_powspec_cylindrical(datacube, kpar_edges, kperp_edges, convolve='True', units='') 
        ## units are mK^2 h^-3 cMpc^3 
                
                        
        savefile_sim = os.path.join(data_dir, 'sim21cm_seed'+str(seedvalue)+'.npz')        
        # np.savez(savefile_data, datacube_21cm = datacube, cylindrical_pow_spec_binned_data = cylindrical_pow_spec_binned_data, kpar_bins = kpar_bins, kperp_bins = kperp_bins, kount = kount)   
        sfile=np.load(savefile_sim)
        cylindrical_pow_spec_binned_sim = sfile['cylindrical_pow_spec_binned_signal']
        kpar_edges=sfile['kpar_edges']
        kperp_edges = sfile['kperp_edges']
        # print(kpar_edges.shape, kperp_edges.shape)
        
        savefile_data = os.path.join(data_dir, 'data21cm_seed'+str(seedvalue)+'.npz')
        sfile=np.load(savefile_data)
        cylindrical_pow_spec_binned_data = sfile['cylindrical_pow_spec_binned_data']

        return cylindrical_pow_spec_binned_sim[0:10,0:10], cylindrical_pow_spec_binned_data[0:10,0:10], kpar_edges[0:10], kperp_edges[0:10]

############ MAIN #######################
seedlist=np.genfromtxt('mockdata_seed_list.txt',dtype=int) ## the first 50 seeds of musicseed_list.txt

seed=seedlist[10]

vmax= 5000
pspec_sim, pspec_data, kpar_edges, kperp_edges = get_21cm_data_seed(seed)


fig = plt.figure(figsize=(12,5))
ax_dim = fig.add_subplot(221)
im_dim = ax_dim.imshow(pspec_sim, origin='lower', vmax=vmax, extent=[kperp_edges.min(), kperp_edges.max(), kpar_edges.min(), kpar_edges.max()])
ax_dim.set_xlabel("$k_{\perp}$ (h/Mpc)")
ax_dim.set_ylabel("$k_{\parallel}$ (h/Mpc)")
ax_dim.set_title(" sim ; seed =  "+str(seed))

# Add colorbar
cbar = fig.colorbar(im_dim, ax=ax_dim,shrink=0.7)
cbar.set_label("$P(k_{\perp}, k_{\parallel}) [mK^2~h^{-3}~cMpc^{3}]$")  # Optional: set label

ax_dim = fig.add_subplot(222)
im_dim = ax_dim.imshow(pspec_data, origin='lower', vmax=vmax,extent=[kperp_edges.min(), kperp_edges.max(), kpar_edges.min(), kpar_edges.max()])
ax_dim.set_xlabel("$k_{\perp}$ (h/Mpc)")
ax_dim.set_ylabel("$k_{\parallel}$ (h/Mpc)")
#ax_dim.set_title("Dimension full power spectrum : seed = "+str(seed))
ax_dim.set_title(" data = sim + noise ; seed =  "+str(seed))

# Add colorbar
cbar = fig.colorbar(im_dim, ax=ax_dim,shrink=0.7)
cbar.set_label("$P(k_{\perp}, k_{\parallel}) [mK^2~h^{-3}~cMpc^{3}]$")  # Optional: set label

###################################################################################################################################

seed=seedlist[30]

#vmax=500
pspec_sim, pspec_data, kpar_edges, kperp_edges = get_21cm_data_seed(seed)
 

#fig = plt.figure(figsize=(12,5))
ax_dim = fig.add_subplot(223)
im_dim = ax_dim.imshow(pspec_sim, origin='lower', vmax=vmax, extent=[kperp_edges.min(), kperp_edges.max(), kpar_edges.min(), kpar_edges.max()])
ax_dim.set_xlabel("$k_{\perp}$ (h/Mpc)")
ax_dim.set_ylabel("$k_{\parallel}$ (h/Mpc)")
ax_dim.set_title(" sim ; seed =  "+str(seed))

# Add colorbar
cbar = fig.colorbar(im_dim, ax=ax_dim,shrink=0.7)
cbar.set_label("$P(k_{\perp}, k_{\parallel}) [mK^2~h^{-3}~cMpc^{3}]$")  # Optional: set label

ax_dim = fig.add_subplot(224)
im_dim = ax_dim.imshow(pspec_data, origin='lower', vmax=vmax,extent=[kperp_edges.min(), kperp_edges.max(), kpar_edges.min(), kpar_edges.max()])
ax_dim.set_xlabel("$k_{\perp}$ (h/Mpc)")
ax_dim.set_ylabel("$k_{\parallel}$ (h/Mpc)")
#ax_dim.set_title("Dimension full power spectrum : seed = "+str(seed))
ax_dim.set_title(" data = sim + noise ; seed =  "+str(seed))
 
# Add colorbar
cbar = fig.colorbar(im_dim, ax=ax_dim,shrink=0.7)
cbar.set_label("$P(k_{\perp}, k_{\parallel}) [mK^2~h^{-3}~cMpc^{3}]$")  # Optional: set label


plt.show()
