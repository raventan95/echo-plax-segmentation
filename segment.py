import os
os.environ['CUDA_VISIBLE_DEVICES']='-1' # comment this out if using GPU
from skimage.transform import resize
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import imageio


input_folder = "./sample/"
model_name = "./model/mymodel_segmentation_1_0.8930.h5"
input_shape = (384,384)
output_folder = "./result/"

if not os.path.exists(output_folder): os.makedirs(output_folder)

def dice_coef(y_true, y_pred):
    smooth = 1 # to avoid division by 0
    y_true = y_true[:,:,:,1:] # ignore background segmentation
    y_pred = y_pred[:,:,:,1:]
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2.*intersection + smooth) / (union + smooth), axis=0) # average across samples in a batch

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
	
videos = [f for f in os.listdir(input_folder) if os.path.isfile(input_folder+f) and f[-3:]=='mp4']
model = load_model(model_name, custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef})

for video in videos:
	print('Segmenting', video, '...')
	images = np.array(imageio.mimread(input_folder+video, memtest=False))
	if len(images.shape) == 4:
		images = images[:,:,:,0] # convert RGB to grayscale
	reader = imageio.get_reader(input_folder+video, 'ffmpeg')
	fps = reader.get_meta_data()['fps']
	
	writer = imageio.get_writer(output_folder+video[:-4]+'_segmented.mp4', fps=fps)
	for i,image in enumerate(images):
		print(i+1,'/', images.shape[0], 'frames')
		image = resize(image,input_shape,preserve_range=True).reshape(1,input_shape[0],input_shape[1],1)
		preds = model.predict(image)[0,:,:,:]
		preds = np.argmax(preds,2)
		
		fig = plt.figure()
		ax = fig.add_axes([0,0,1,1])
		ax.axis('off')
		plt.imshow(images[i,:,:], cmap='gray')
		plt.imshow(resize(preds,(images.shape[1],images.shape[2])), cmap='nipy_spectral', alpha=0.25)
		plt.savefig('./overlay.png', bbox_inches=0)
		plt.close()
		overlay = imageio.imread('./overlay.png')
		writer.append_data(overlay)
	writer.close()
	os.remove('./overlay.png')
print('Segmentation complete.')