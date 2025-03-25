import os
import numpy as np
from tqdm import tqdm
from script import two_lpt
import script

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
            nbins = int( (kmax - kmin) / dk) 
        else:
            dk = (kmax - kmin) / (nbins)
        k_edges = np.linspace(kmin, kmax, num=nbins+1, endpoint=True)
        k_bins = (k_edges[:-1] + k_edges[1:]) / 2

    return k_edges, k_bins
    
    
def get_21cm_powspec_seed(seedvalue):

        ## Create an array of redshift values where snapshots will be created.

        data_dir = './simulation_files/files_seed'+str(seedvalue)
        data_path = data_dir+'/snapshots_2lpt/'
        data_root = 'snap'
        outpath = os.path.join(data_dir, 'script_files')  # Directory where script files are stored   

        ## script parameters

        ngrid=128
        box = 256.0 ### Mpc/h
        scaledist = 1.e-3
        
        ## cosmology parameters
                        
        omega_m = 0.308
        omega_l = 1 - omega_m
        omega_b = 0.0482
        h = 0.678
        sigma_8 = 0.829
        ns = 0.961
        
        ## simulation parameters
        
        log10Mmin=9.0
        zeta=15    
        z = 7.0
        snapshot=0
        
        Tbar = 27 * ((1 + z) / 10) ** 0.5 * (0.15 / (omega_m * h ** 2)) ** 0.5 ## units = mK , ignoring the omega_b dependence
        box_cMpc = box/h
        gadget_snap=f"{data_path}/{data_root}_{snapshot:03}"

        default_simulation_data = script.default_simulation_data(gadget_snap, outpath, sigma_8=0.829, ns=0.961, omega_b=0.0482, scaledist=scaledist)

        matter_fields = script.matter_fields(default_simulation_data, ngrid, outpath, overwrite_files=False)

        ionization_map = script.ionization_map(matter_fields)

        kpar_edges,kpar_bins = get_k_edges(boxlength = box , numgrid = ngrid)           ## units : h/cMpc
        kperp_edges,kperp_bins = get_k_edges(boxlength = box , numgrid = ngrid)         ## units : h/cMpc        
        
        '''
        Kpar, Kperp = np.meshgrid(kpar_bins, kperp_bins)
        k_bins = np.vstack([Kperp.ravel(), Kpar.ravel()])
        k_values = np.sqrt(k_bins[0]**2 + k_bins[1]**2).reshape(Kpar.shape)   
        '''
        
        #  collapsed fraction and ionized fraction
        fcoll_arr = matter_fields.get_fcoll_for_Mmin(log10Mmin)
            
        qi_arr = ionization_map.get_qi(zeta * fcoll_arr)

        # mass-weighted neutral hydrogen and brightness temperature  
                  
        Delta_HI_arr = (1 - qi_arr) * (1 + matter_fields.densitycontr_arr) 
        
        Delta_Tb_arr =   Tbar * Delta_HI_arr                                            ## units : mK
        
        matter_fields.initialize_powspec()
        
        cylindrical_pow_spec_binned, kount = ionization_map.get_binned_powspec_cylindrical(Delta_Tb_arr, kpar_edges, kperp_edges, convolve='True', units='') 
        ## units are mK^2 h^-3 cMpc^3 
        
        cylindrical_pow_spec_binned2, _ = ionization_map.get_binned_powspec_cylindrical(Delta_HI_arr, kpar_edges, kperp_edges, convolve='True', units='mK')

        return cylindrical_pow_spec_binned, cylindrical_pow_spec_binned2, kpar_edges, kperp_edges

############ MAIN #######################
seedlist=np.genfromtxt('musicseed_list.txt',dtype=int)
seed=seedlist[0]

vmax = 500
import matplotlib.pyplot as plt 
pspec1, pspec2, kpar_edges, kperp_edges = get_21cm_powspec_seed(seed)


fig = plt.figure(figsize=(12,5))
ax_dim = fig.add_subplot(121)
im_dim = ax_dim.imshow(pspec1, origin='lower', vmax=vmax, extent=[kperp_edges.min(), kperp_edges.max(), kpar_edges.min(), kpar_edges.max()])
ax_dim.set_xlabel("$k_{\perp}$ (h/Mpc)")
ax_dim.set_ylabel("$k_{\parallel}$ (h/Mpc)")
ax_dim.set_title("external Tb multiplication : seed = "+str(seed))

# Add colorbar
cbar = fig.colorbar(im_dim, ax=ax_dim,shrink=0.7)
cbar.set_label("$P(k_{\perp}, k_{\parallel}) [mK^2~h^{-3}~cMpc^{3}]$")  # Optional: set label



ax_dim = fig.add_subplot(122)

im_dim = ax_dim.imshow(pspec2, origin='lower', vmax=vmax,extent=[kperp_edges.min(), kperp_edges.max(), kpar_edges.min(), kpar_edges.max()])
ax_dim.set_xlabel("$k_{\perp}$ (h/Mpc)")
ax_dim.set_ylabel("$k_{\parallel}$ (h/Mpc)")
ax_dim.set_title("internal Tb multiplication : seed = "+str(seed))

# Add colorbar
cbar = fig.colorbar(im_dim, ax=ax_dim,shrink=0.7)
cbar.set_label("$P(k_{\perp}, k_{\parallel}) [mK^2~h^{-3}~cMpc^{3}]$")  # Optional: set label

plt.show()

