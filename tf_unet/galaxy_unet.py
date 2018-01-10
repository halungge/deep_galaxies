from __future__ import division, print_function

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
import wget
import time
import os
plt.rcParams['image.cmap'] = 'gist_earth'

from scripts.galaxy_util import DataProvider
from tf_unet import unet

base_path = "C:/Users/user/Desktop/deep_galaxies/data/"
output_path="../output/"
print(glob.glob(os.path.curdir))

from tf_unet import unet
from scripts import galaxy_util
data_provider = galaxy_util.DataProvider(200,200, base_path + "*.fits", data_suffix=".fits", mask_suffix='.star_seg.fits', mask_suffix2='.gal_seg.fits', mask_suffix3='.comb_seg.fits',shuffle_data=True, n_class=2)

net = unet.Unet(layers=3, features_root=16, channels=1, n_class=2)

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.5)) #momentum=0.2 previously
path = trainer.train(data_provider, out_dir,
                     training_iters=32,
                     epochs=3,
                     dropout=0.1,
                     display_step=2)#previously dropout=0.5

##tensorboard --logdir runs  #im Terminal im richtigen Verzeichnis eingeben -> dann angegebene Website im Chrome öffnen