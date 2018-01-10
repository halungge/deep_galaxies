from tf_unet import unet, util, image_util
from astropy.io import fits

base_path = "./data/data1/"
output_path="./output/"

#preparing data loading
# Load fits file containing image data
data_provider = image_util.FitsImageDataProvider(base_path + "/*.fits", data_suffix=".fits",
                                                 mask_suffix='.gal_seg.fits', 
                                                 shuffle_data=True, n_class=2)

#setup & training
net = unet.Unet(layers=1, features_root=4, channels=1, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, output_path, training_iters=2, epochs=1)

#verification
...
x_test, y_test = data_provider(1)
prediction = net.predict(path, x_test)

unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))

img = util.combine_img_prediction(x_test, y_test, prediction)
util.save_image(img, "prediction.jpg")