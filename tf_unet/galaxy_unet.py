from tf_unet import unet, image_util, image_util
import time
import os
from scripts.galaxy_util import DataProvider
from astropy.io import fits

base_path = "../data/data1/"
output_path="../output/"

#preparing data loading
# Load fits file containing image data
#data_provider = image_util.FitsImageDataProvider(base_path + "/*.fits", data_suffix=".fits",
 #                                                mask_suffix='.gal_seg.fits',
 #                                                shuffle_data=True, n_class=2)
data_provider = DataProvider(200,200, base_path + "*.fits", data_suffix=".fits",
                                                mask_suffix='.gal_seg.fits',
                                               shuffle_data=True, n_class=2)
print(type(data_provider))
#setup & training
net = unet.Unet(layers=3, features_root=16, channels=1, n_class=2)
trainer = unet.Trainer(net)

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
pred_path = os.path.abspath(os.path.join(os.path.curdir, "runs2", timestamp))

#default: train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False, write_graph=False, prediction_path = 'prediction')
path = trainer.train(data_provider, out_dir, training_iters=20, epochs=100, display_step=2, write_graph=True,prediction_path=pred_path)
print(out_dir)
#verification
x_test, y_test = data_provider(1)
prediction = net.predict(path, x_test)

unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))

img = util.combine_img_prediction(x_test, y_test, prediction)
util.save_image(img, "prediction.jpg")
#tensorboard --logdir runs


