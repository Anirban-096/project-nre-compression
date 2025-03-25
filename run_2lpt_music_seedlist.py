###
# This example shows how to run MUSIC to generate 2LPT particle positions
# and velocities that are needed for ionization maps.
###

import os
import numpy as np
from tqdm import tqdm
from script import two_lpt

## The full path to the MUSIC executable.
#music_exec = "../music/MUSIC"
music_exec = "/data/anirban/softwares/music/MUSIC" #set the MUSIC executable path
    
def run_MUSIC(seedvalue):    
    ## box size
    box = 256.0 ### Mpc/h

    ## Create an array of redshift values where snapshots will be created.

    #alist = np.linspace(0.0625, 0.1667, num=51) ## equally spaced list of scale factor values between z = 15 and 5
    #zlist = 1 / alist - 1
    #zlist = np.linspace(5.0, 15.0, num=51)

    zlist = np.array([7.0])

    outpath = './simulation_files/files_seed'+str(seedvalue)+'/snapshots_2lpt' ## output directory

    os.makedirs(outpath, exist_ok=True)

    outroot = 'snap' ## root of the output snapshots
    dx = 1. ## the grid resolution in Mpc/h, is also the mean inter-particle distance

    ## cosmological parameters
    omega_m = 0.308
    omega_l = 1 - omega_m
    omega_b = 0.0482
    h = 0.678
    sigma_8 = 0.829
    ns = 0.961

    two_lpt.run_music(music_exec,
                  box,
                  zlist,
                  seed,
                  outpath,
                  outroot,
                  dx,
                  omega_m,
                  omega_l,
                  omega_b,
                  h,
                  sigma_8,
                  ns,)


seedlist=np.genfromtxt('musicseed_list.txt',dtype=int)

for i, seed in enumerate(tqdm(seedlist, desc="Running MUSIC")):
    run_MUSIC(seed)

