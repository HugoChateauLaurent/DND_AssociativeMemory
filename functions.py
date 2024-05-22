import numpy as np 
from copy import deepcopy
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(['nature'])

USE_THRESHOLD = False

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                  "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]) 


### image transformation functions ###
# function to half the mnist image to use as a probe query
"""
def halve_continuous_img(img,sigma=None):
	# mnist
	if len(img) == (28 * 28):
		H,W = img.reshape(28,28).shape
		i = deepcopy(img.reshape(28,28))
		i[H//2:H,:] = 0
		return i
	# cifar -- this is a bit of a hack doing it like this
	elif len(img) == (32 * 32 * 3):
		C,H,W = img.reshape(3,32,32).shape
		i = deepcopy(img.reshape(3,32,32))
		i[:,H//2:H,:] = 0
		return i
	else:
		raise ValueError("Input data dimensions not recognized")
""" 
def halve_continuous_img(img, sigma=None,reversed = False):
    return mask_continuous_img(img, 0.5, reversed = reversed)

def mask_continuous_img(img, frac_masked,reversed = False):
	# mnist
	frac_masked = 1-frac_masked
	if len(img) == (28*28):
		i = deepcopy(img.reshape(28,28))
		H,W = i.shape
		if reversed:
			i[0:int(H * frac_masked),:] = 0
		else:
			i[int(H * frac_masked):H,:] = 0
		return i
	elif len(img) == (32 * 32 * 3):
		i = deepcopy(img.reshape(3,32,32))
		C,H,W = i.shape
		if reversed:
			i[:,0:int(H*frac_masked),:] = 0
		else:
			i[:,int(H*frac_masked):H,:] = 0
		return i
	# imagenet
	elif len(img) == (64 * 64 * 3):
		i = deepcopy(img.reshape(3,64,64))
		C,H,W = i.shape
		if reversed:
			i[:,0:int(H*frac_masked),:] = 0
		else:
			i[:,int(H*frac_masked):H,:] = 0
		return i
	else:
		raise ValueError("Input data dimensions not recognized")

def gaussian_perturb_image(img, sigma=0.1):
	#print(img.shape)
	if len(img.shape) != 1:
		total_img_len = np.prod(np.array(img.shape))
		img = img.reshape(total_img_len)
	N = len(img)
	variance = torch.tensor(np.identity(N) * sigma).float()
	perturb = torch.normal(0,sigma,size=[N,])
	return torch.clamp(torch.abs(img + perturb),0,1)

def random_mask_frac(img, mask_prob):
    img_shape = img.shape
    flat_img = deepcopy(img).flatten()
    for i in range(len(flat_img)):
        r = np.random.uniform(0,1)
        if r <= mask_prob:
            flat_img[i] = 0.0
    return flat_img.reshape(img_shape)

def random_mask_frac_handle_color(img, mask_prob):
	img_shape = img.shape
	if len(img) == 28*28:
		return random_mask_frac(img, mask_prob)
	elif len(img) == 32*32*3:
		reshp = deepcopy(img).reshape(3,32,32)
		for i in range(32):
			for j in range(32):
				r = np.random.uniform(0,1)
				if r <= mask_prob:
					reshp[:,i,j] = 0
		return reshp.reshape(img_shape)
	elif len(img) == 64*64*3:
		reshp = deepcopy(img).reshape(3,64,64)
		for i in range(64):
			for j in range(64):
				r = np.random.uniform(0,1)
				if r <= mask_prob:
					reshp[:,i,j] = 0
		return reshp.reshape(img_shape)
	else:
		raise ValueError("image shape not recognized")
					

def image_inpaint(img, mask_frac):
    #pixels_to_mask = 
	if len(img) == (28*28):
		i = deepcopy(img.reshape(28,28))
		H,W = i.shape
		pixels_to_mask = int(H * mask_frac // 2)
		i[0:pixels_to_mask,:] = 0
		i[28-pixels_to_mask:28,:] = 0
		i[:, 0:pixels_to_mask] = 0
		i[:, 28 - pixels_to_mask:28] = 0
		return i
	elif len(img) == (32 * 32 * 3):
		i = deepcopy(img.reshape(3,32,32))
		C,H,W = i.shape
		pixels_to_mask = int(H * mask_frac // 2)
		i[:,0:pixels_to_mask,:] = 0
		i[:,32-pixels_to_mask:32,:] = 0
		i[:,:, 0:pixels_to_mask] = 0
		i[:,:, 32 - pixels_to_mask:32] = 0
		return i
	# imagenet
	elif len(img) == (64 * 64 * 3):
		i = deepcopy(img.reshape(3,64,64))
		C,H,W = i.shape
		pixels_to_mask = int(H * mask_frac // 2)
		i[:,0:pixels_to_mask,:] = 0
		i[:,64-pixels_to_mask:64,:] = 0
		i[:,:, 0:pixels_to_mask] = 0
		i[:,:, 64 - pixels_to_mask:64] = 0
		return i
	else:
		raise ValueError("Input data dimensions not recognized")

def binary_to_bipolar(x):
    return torch.sign(x - 0.5)

def bipolar_to_binary(x):
  	return (x + 1) /2

### update functions ###

EPSILON = 1e-4
# update rule of the Modern Continuous Hopfield Network which is closely related to attention etc. Computes similarity scores according in the dot-product or cosine similarity space
def MCHN_update_rule(X,z,beta, norm=True):
	out = X.T @ F.softmax(beta * X @ z,dim=0)
	if norm:
		return out / torch.sum(out)
	else:
		return out

# PC associative memory update rule -- Computes similarity scores in the euclidean distance/ prediction error space
def PC_update_rule(X,z,beta,f = torch.square):
	e = z - X # compute prediction error
	return X.T @ F.softmax(beta * -torch.sum(f(e), axis=1))

def dot_product_distance(X,z):
	return X @ z.reshape(len(z[0,:]))

def normalized_dot_product(X,z):
	norm_X = X / torch.sum(X, axis=1).reshape(X.shape[0],1)
	norm_z = z / torch.sum(z)
	dots = norm_X @ norm_z.reshape(len(z[0,:]))
	recip = dots
	norm_dot = recip / torch.sum(recip)
	return norm_dot


def cosine_similarity(X,z):
	return (X @ z.reshape(len(z[0,:]))) / (torch.norm(X) * torch.norm(z))

def manhattan_distance(X,z):
	return 1/(torch.sum(torch.abs(z - X),axis=1)+EPSILON)
	#return -torch.sum(torch.abs(z - X), axis=1)

def euclidean_distance(X,z):
	return 1/(torch.sum(torch.square(z - X),axis=1)+EPSILON)
	#return -torch.sum(torch.square(z - X),axis=1)

def general_update_rule(X,z,beta,sim, sep=F.softmax,sep_param=1,norm=True):
	sim_score = beta * sim(X,z)

	#print("SIMS: ", sim_score)
	if norm:
		sim_score = sim_score / torch.sum(sim_score)
	sep_score = sep(sim_score,sep_param)
	if norm:
		sep_score = sep_score / torch.sum(sep_score)
	#print("AUTO SEP SCORE: ", sep_score)
	out = X.T @ sep_score
	return out


def heteroassociative_update_rule(M,P,z,beta, sim,sep=F.softmax, sep_param=1, norm=True):
	sim_score = beta * sim(M,z)
	if norm:
		sim_score = sim_score / torch.sum(sim_score)
	#print(sim_score.shape)
	sep_score = sep(sim_score, sep_param)
	#print(sep_score.shape)
	if norm:
		sep_score = sep_score / torch.sum(sep_score)
	print(sep_score)
	out = P.T @ sep_score
	return out

# inefficient implementation but works well enough
def KL_divergence(X,z):
	KL_matrix = torch.zeros_like(X)
	for i in range(len(X)):
		x = X[i,:]
		x_norm = x / torch.sum(x)
		z_norm = z / torch.sum(z)
		KL_matrix[i,:] = x_norm * (torch.log(x_norm + EPSILON) - torch.log(z_norm + EPSILON))
	return 1/(torch.sum(KL_matrix,axis=1)+EPSILON)

def reverse_KL_divergence(X,z):
	KL_matrix = torch.zeros_like(X)
	for i in range(len(X)):
		x = X[i,:]
		x_norm = x / torch.sum(x)
		z_norm = z / torch.sum(z)
		KL_matrix[i,:] = z_norm* (torch.log(z_norm + EPSILON) - torch.log(x_norm + EPSILON))
	return 1/(torch.sum(KL_matrix,axis=1)+EPSILON)

def Jensen_Shannon_divergence(X,z):
	return 1/((0.5 * KL_divergence(X,z) + 0.5 * reverse_KL_divergence(X,z))+EPSILON)

### potential separation functions --  linear, sublinear (sqrt, log), polynomial, exponential, max ###
def separation_log(x, param):
	return torch.log(x)

def separation_identity(x,param):
	return x

def separation_softmax(x,param):
	return F.softmax(param * x, dim=0) # param = beta = softmax temperature

def separation_softmax_beta10(x,param):
	return separation_softmax(x,10)

def separation_softmax_beta100(x,param):
	return separation_softmax(x,100)

def separation_softmax_beta1000(x,param):
	return separation_softmax(x,1000)

def separation_softmax_beta10000(x,param):
	return separation_softmax(x,10000)

def separation_softmax_beta100000(x,param):
	return separation_softmax(x,100000)

def separation_polynomial(x,param):	
    return torch.pow(x, param)

def separation_square(x,param):
	return separation_polynomial(x,2)

def separation_cube(x,param):
	return separation_polynomial(x,3)

def separation_sqrt(x,param):
	return separation_polynomial(x,0.5)

def separation_quartic(x,param):
	return separation_polynomial(x,4)

def separation_ten(x,param):
	return separation_polynomial(x,10)

def separation_kmax(x, param, norm_f=lambda x: x/torch.sum(x)): # param = k of kNN
	param = min(param, torch.numel(x))
	kNN_ind = torch.topk(x, param).indices # find NN
	mask = torch.ones_like(x, dtype=bool)
	mask[kNN_ind] = False
	x[mask] *= 0
	x = norm_f(x) # normalize over NN
	return x

def separation_max(x, param):
	z = separation_kmax(x, 1)
	return z

def separation_2max(x, param):
	z = separation_kmax(x, 2)
	return z

def separation_5max(x, param):
	z = separation_kmax(x, 5)
	return z

def separation_10max(x, param):
	z = separation_kmax(x, 10)
	return z

def separation_50max(x, param):
	z = separation_kmax(x, 50)
	return z

def separation_2softmax(x, param):
	z = separation_kmax(x, 2, norm_f=F.softmax)
	return z

def separation_5softmax(x, param):
	z = separation_kmax(x, 5, norm_f=F.softmax)
	return z

def separation_10softmax(x, param):
	z = separation_kmax(x, 10, norm_f=F.softmax)
	return z

def separation_50softmax(x, param):
	z = separation_kmax(x, 50, norm_f=F.softmax)
	return z



# function to iterate through the images, retrieve the output and compute the amount of correctly retrieved image queries
def reshape_img_list(imglist, imglen, opt_fn = None):
	new_imglist = torch.zeros(len(imglist), imglen)
	for i,img in enumerate(imglist):
		img = img.reshape(imglen)
		if opt_fn is not None:
			img = opt_fn(img)
		new_imglist[i,:] = img
	return new_imglist

# key functions which actually tests the storage capacity of the associative memory
def PC_retrieve_store_continuous(imgs,N, P = None, beta=1,num_plot = 5,similarity_metric="error",f=manhattan_distance, image_perturb_fn = halve_continuous_img,sigma=0.5,sep_fn=separation_max, sep_param=1, use_norm = True,ERROR_THRESHOLD = 60, network_type="", return_sqdiff_outputs = False, plot_example_reconstructions = False):
	X = imgs[0:N,:]
	img_len = np.prod(np.array(X[0].shape))
	if len(X.shape) != 2:
		if network_type == "classical_hopfield":
			X = reshape_img_list(X, img_len, opt_fn = binary_to_bipolar)
		else:
			X = reshape_img_list(X, img_len)
	N_correct = 0
	#ERROR_THRESHOLD = 80
	for j in range(N):
		if network_type == "classical_hopfield":
			z = binary_to_bipolar(image_perturb_fn(X[j,:],sigma)).reshape(1, img_len)
		else:
			z = image_perturb_fn(X[j,:],sigma).reshape(1,img_len)
		if P is None: # autoassociative
			out = general_update_rule(X,z,beta, f,sep=sep_fn, sep_param=sep_param,norm=use_norm).reshape(img_len)
			if network_type == "classical_hopfield":
				out = binary_to_bipolar(torch.sign(out))
			sqdiff = torch.sum(torch.square(X[j,:] - out))
			if plot_example_reconstructions:
				plt.imshow(X[j,:].reshape(3,32,32).permute(1,2,0))
				plt.show()
				plt.imshow(z.reshape(3,32,32).permute(1,2,0))
				plt.show()
				plt.imshow(out.reshape(3,32,32).permute(1,2,0))
				plt.show()
				print("SQDIFF: ", sqdiff)
		else: # heteroassociative
			P = P[0:N,:]
			out = heteroassociative_update_rule(X,P,z,beta, f,sep=sep_fn, sep_param=sep_param, norm=use_norm).reshape(img_len)
			if network_type == "classical_hopfield":
				out = binary_to_bipolar(torch.sign(out))
			sqdiff = torch.sum(torch.square(P[j,:] - out))
			print("sqdiff hetero: ", sqdiff)
		if torch.abs(sqdiff) <= ERROR_THRESHOLD and USE_THRESHOLD:
			N_correct +=1
		if not USE_THRESHOLD:
			if P is None:
				if sqdiff <= torch.sum(torch.square(X - out[None,:]), axis=-1).min():
					N_correct += 1
			else:
				print("ERROR: heteroassociation without threshold not implemented")
				assert False
		if j < num_plot:
			fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
			titles = ["Original","Masked","Reconstruction"]
			plot_list = [X[j,:], z, out]
			for i, ax in enumerate(axs.flatten()):
				plt.sca(ax)
				#print(imgs[0].shape)
				if len(imgs[0].shape) == 3:
					plt.imshow(plot_list[i].reshape(imgs[0].shape).permute(1,2,0))
				else:
					plt.imshow(plot_list[i].reshape(28,28))
				plt.title(titles[i])
			plt.show()
	return N_correct / N


if __name__ == '__main__':
	from data import *
	trainset_cifar, testset_cifar = get_cifar10(10000)
	imgs = trainset_cifar[0][0]
	xs = [imgs[i,:].reshape(np.prod(imgs[i,:].shape)) for i in range(100)]
	plt.subplot(1,2,1)
	plt.imshow(xs[0].reshape(imgs[0,:].shape).permute(1,2,0))
	plt.title("Original Image")
	halved_img = halve_continuous_img(xs[0])
	plt.subplot(1,2,2)
	plt.title("Masked Image")
	plt.imshow(halved_img.permute(1,2,0))
	plt.show()
	gauss_img = gaussian_perturb_image(xs[0],0.1).reshape(3,32,32)
	plt.subplot(1,2,2)
	plt.title("Gaussian Image")
	plt.imshow(gauss_img.permute(1,2,0))
	plt.show()
	mask_fracs = [0.1,0.3,0.5,0.7,0.9]
	for frac in mask_fracs:
		mask_img = mask_continuous_img(xs[0],frac).reshape(3,32,32)
		plt.subplot(1,2,2)
		plt.title("Masked Image")
		plt.imshow(mask_img.permute(1,2,0))
	plt.show()