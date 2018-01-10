# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 18, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import h5py
import numpy as np
from astropy.io import fits
from tf_unet.image_util import ImageDataProvider


class DataProvider(ImageDataProvider):
    """
    Extends the BaseDataProvider to randomly select the next 
    data chunk
    """

    channels = 1
    n_class = 2
    # def __init__(self, nx, files, a_min=30, a_max=210):
    #     super(DataProvider, self).__init__(a_min, a_max)
    #     self.nx = nx
    #     self.files = files
    #
    #     assert len(files) > 0, "No training files"
    #     print("Number of files used: %s"%len(files))
    #     self._cylce_file()

    def __init__(self, nx , ny, search_path, a_min=None, a_max=None, data_suffix=".fits", mask_suffix='_mask.fits', mask_suffix2='_mask.tif', mask_suffix3='_mask.tif',
                     shuffle_data=True, n_class=2):
            super(ImageDataProvider, self).__init__(a_min, a_max)
            self.nx=nx
            self.ny=ny
            self.data_suffix = data_suffix
            self.mask_suffix = mask_suffix
            self.mask_suffix2 = mask_suffix2
            self.mask_suffix3 = mask_suffix3

            self.file_idx = -1
            self.shuffle_data = shuffle_data
            self.n_class = n_class

            self.data_files = self._find_data_files(search_path)
            print(search_path)
            print(self.data_files)
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

            assert len(self.data_files) > 0, "no training files"
            print("Number of files used: %s" % len(self.data_files))

            img = self._load_file(self.data_files[0])
            self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _load_file(self, path, dtype=np.float32):
            image = fits.getdata(path)
            return image
            # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    # def _read_chunck(self):
    #     with h5py.File(self.files[self.file_idx], "r") as fp:
    #         nx = fp["data"].shape[1]
    #         idx = np.random.randint(0, nx - self.nx)
    #
    #         sl = slice(idx, (idx+self.nx))
    #         data = fp["data"][:, sl]
    #         gal = fp["mask"][:, sl]

    def _read_chunck(self):
        imgname = self.data_files[self.file_idx]
        img = self._load_file(imgname)
        M = img.shape[0]
        N = img.shape[1]
        idx1 = np.random.randint(0, M - self.nx)
        idx2 = np.random.randint(0, N - self.ny)
        #sl1=slice(idx1,(idx1+self.nx))
        #sl2= slice(idx2,idx2+self.ny)

        data=img[idx1:(idx1+self.nx),idx2:idx2+self.ny]

        #data=img[sl1,sl2]


        label_name = imgname.replace(self.data_suffix, self.mask_suffix)
        print(label_name)
        label = fits.getdata(label_name)

        #gal = label[sl1,sl2];
        gal = label[idx1:(idx1+self.nx),idx2:idx2+self.ny]

        return data, gal
    
    def _next_data(self):
        data, gal = self._read_chunck()
        nx = data.shape[1]
        while nx < self.nx:
            self._cylce_file()
            data, gal = self._read_chunck()
            nx = data.shape[1]
            
        return data, gal

    def _cylce_file(self):
        self.file_idx = np.random.choice(len(self.files))
        
