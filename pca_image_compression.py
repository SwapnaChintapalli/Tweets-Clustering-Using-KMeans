import numpy as np
from PIL import Image
from scipy.misc import imsave
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-i', '--image', required=True, help="Path to input image")

args = vars(parser.parse_args())

a = Image.open(args['image'])
width, height = a.size
print(width, height)

file_path = args['image'].split('/')

file_name = file_path[-1]

a_np = np.array(a)
a_r = a_np[:,:,0]
a_g = a_np[:,:,1]
a_b = a_np[:,:,2]


def pca(image_2d, numpc):
	cov_mat = image_2d - np.mean(image_2d , axis = None)
	eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat))
	p = np.size(eig_vec, axis =None)
	idx = np.argsort(eig_val)
	idx = idx[::-1]
	eig_vec = eig_vec[:,idx]
	if numpc <p or numpc >0:
		eig_vec = eig_vec[:, range(numpc)]
	score = np.dot(eig_vec.T, cov_mat)
	recon = np.dot(eig_vec, score) + np.mean(image_2d, axis = None).T
	reconstructed_image = np.uint8(np.absolute(recon))
	return reconstructed_image

red = []
green = []
blue = []

for i in range (0,3):
    numpc = (i*30)+30
    red.append(pca(a_r, numpc))
    green.append(pca(a_g, numpc))
    blue.append(pca(a_b, numpc))
    recon_color_img = np.dstack((red[i], green[i], blue[i]))
    recon_color_img = Image.fromarray(recon_color_img)
    # recon_color_img.show()
    imsave("./compressedImages/" + file_name.split('.')[0] + "_" + str(i+1) + "_compressed.jpg", recon_color_img)
