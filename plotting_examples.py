# functions for plotting graphs given the associative memory networks
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(['nature'])

COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
        "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color = COLORS) 

import seaborn as sns
import torch
import numpy as np
from functions import *
from data import *
from copy import deepcopy
import pickle

SAVE_FORMAT = "pdf"
ALPHA = .2

def parse_sname_for_title(sname):
    if "tiny_" in sname or "imagenet_" in sname:
        return "ImageNet"
    if "mnist_" in sname:
        return "MNIST"
    else:
        return "CIFAR10"
    
def generate_demonstration_reconstructions(imgs, N, f=manhattan_distance, image_perturb_fn = mask_continuous_img, perturb_vals =[], sep_fns = [], sep_labels=[], sep_param=1, use_norm=True,sname=""):
    X = imgs[0:N,:]
    img_shape = X[0].shape
    img_len = np.prod(np.array(img_shape))
    if len(img_shape) != 1:
        X = reshape_img_list(X, img_len)
    img_idx = 0 # int(np.random.choice(N))
    init_img = deepcopy(X[img_idx,:])
    perturbed_imgs = []
    reconstructed_imgs = []
    beta = 1
    for val in perturb_vals:
        reconstructed_imgs.append([])
        query_img = image_perturb_fn(init_img, val).reshape(1, img_len)
        perturbed_imgs.append(deepcopy(query_img.reshape(img_shape).permute(1,2,0)))
        for sep_fn in sep_fns:
            out = general_update_rule(X,query_img,beta, f,sep=sep_fn, sep_param=sep_param,norm=use_norm).reshape(img_len)
            reconstructed_imgs[-1].append(deepcopy(out).reshape(img_shape).permute(1,2,0))

    N_vals = len(perturb_vals)
    ncol = N_vals
    nrow = 1 + len(sep_fns)
    fig, ax_array = plt.subplots(nrow, ncol, figsize=(ncol+1,nrow+1), gridspec_kw = {'wspace':0, 'hspace':0, 'top':1.-0.5/(nrow+1), 'bottom': 0.5/(nrow+1), 'left': 0.5/(ncol+1), 'right' :1-0.5/(ncol+1)})
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            if i == 0:
                if perturbed_imgs[j].shape[-1]==1:
                    axes.matshow(perturbed_imgs[j], cmap="inferno")
                else:
                    axes.imshow(perturbed_imgs[j])
                axes.set_title(str(perturb_vals[j]), fontsize=12)
            else:
                if reconstructed_imgs[j][i-1].shape[-1]==1:
                    axes.matshow(reconstructed_imgs[j][i-1], cmap="inferno")
                else:
                    axes.imshow(reconstructed_imgs[j][i-1])
            if j==0:
                axes.set_ylabel((["Cue"]+sep_labels)[i], fontsize=12)
            #axes.set_aspect("auto")
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    fig.suptitle("Noise Variance", fontsize=12)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()

    plt.savefig("figures/"+sname, format=SAVE_FORMAT,bbox_inches = "tight", pad_inches = 0)
    plt.close()

def generate_demonstration_reconstructions_betas(imgs, N, f=manhattan_distance, image_perturb_fn = mask_continuous_img, perturb_vals =[], betas=[], sep_fn=separation_softmax, sep_param=1, use_norm=True,sname=""):
    X = imgs[0:N,:]
    img_shape = X[0].shape
    img_len = np.prod(np.array(img_shape))
    if len(img_shape) != 1:
        X = reshape_img_list(X, img_len)
    img_idx = 0 # int(np.random.choice(N))
    init_img = deepcopy(X[img_idx,:])
    perturbed_imgs = []
    reconstructed_imgs = []
    for val in perturb_vals:
        reconstructed_imgs.append([])
        query_img = image_perturb_fn(init_img, val).reshape(1, img_len)
        perturbed_imgs.append(deepcopy(query_img.reshape(img_shape).permute(1,2,0)))
        for beta in betas:
            out = general_update_rule(X,query_img,1, f,sep=sep_fn, sep_param=beta,norm=use_norm).reshape(img_len)
            reconstructed_imgs[-1].append(deepcopy(out).reshape(img_shape).permute(1,2,0))

    N_vals = len(perturb_vals)
    ncol = N_vals
    nrow = 1 + len(betas)
    fig, ax_array = plt.subplots(nrow, ncol, figsize=(ncol/3,nrow/3), gridspec_kw = {'wspace':0, 'hspace':0, 'top':1.-0.5/(nrow+1), 'bottom': 0.5/(nrow+1), 'left': 0.5/(ncol+1), 'right' :1-0.5/(ncol+1)})
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            if i == 0:
                axes.imshow(perturbed_imgs[j])
                axes.set_title(str(perturb_vals[j]))
            else:
                axes.imshow(reconstructed_imgs[j][i-1])
            if j==0:
                axes.set_ylabel((["Cue"]+betas)[i])
            #axes.set_aspect("auto")
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    fig.suptitle("Noise Variance")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    plt.savefig("figures/"+sname, format=SAVE_FORMAT,bbox_inches = "tight", pad_inches = 0)
    plt.close()

def run_example_reconstructions(imgs, dataset_str):
    sigmas = [0.1,0.2,0.3,0.5,0.8,1,1.5]
    # sep_fns = [separation_max, separation_2max, separation_5max, separation_10max, separation_50max, separation_identity]
    # sep_labels = ["Max", "2-Max", "5-Max","10-Max", "50-Max", "Identity"]
    sep_fns = [separation_max, separation_5max, separation_50max]
    sep_labels = ["Max", "5-Max", "50-Max"]
    # sep_softmaxs = [separation_softmax_beta100000, separation_softmax_beta10000, separation_softmax_beta1000, separation_softmax_beta100, separation_softmax]
    # softmaxs_labels = [r"$\beta=100000$", r"$\beta=10000$", r"$\beta=1000$", r"$\beta=100$", r"$\beta=1$"]
    generate_demonstration_reconstructions(
        imgs, 
        100, 
        perturb_vals=sigmas, 
        image_perturb_fn=gaussian_perturb_image, 
        f=euclidean_distance, 
        sep_fns=sep_fns,
        sep_labels=sep_labels,
        sname="euclidean_{dataset_str}_examples.".format(dataset_str=dataset_str) + SAVE_FORMAT
    )
    # generate_demonstration_reconstructions(
    #     imgs, 
    #     100, 
    #     perturb_vals=sigmas, 
    #     image_perturb_fn=gaussian_perturb_image, 
    #     f=euclidean_distance, 
    #     sep_fns=sep_softmaxs,
    #     sep_labels=softmaxs_labels,
    #     sname="euclidean_{dataset_str}_betas_examples.".format(dataset_str=dataset_str) + SAVE_FORMAT
    # )
    
if __name__ == '__main__':

    #dataset_str = "mnist_longer_capacity_"
    dataset_str = "cifar_10_"
    #dataset_str = ""
    dataset_str = "mnist_"
    dataset_str = "tiny_"
    #separation functions
    #dataset_str = "imagenet_"


    for dataset_str in ["mnist_", "cifar_10_", "tiny_"]:

        print(dataset_str)

        LOAD_DATA = False
        PLOT_RESULTS = True
        imgs = []
        if not LOAD_DATA:
            if dataset_str == "tiny_" or dataset_str == "imagenet_" or dataset_str == "tiny_2_":
                imgs = load_tiny_imagenet(N_imgs=10000)
                imgs = torch.swapaxes(imgs, -1, 1)
            if dataset_str == "mnist_" or dataset_str == "mnist_longer_capacity_":
                trainset_mnist, testset_mnist = load_mnist(60000)
                imgs = trainset_mnist[0][0]
            if dataset_str == "" or dataset_str == "cifar_10_" or dataset_str == "cifar":
                trainset_cifar, testset_cifar = get_cifar10(10000)
                imgs = trainset_cifar[0][0]


        print(dataset_str, imgs.shape)
        # example reconstructions (Fig. 5?)
        run_example_reconstructions(imgs, dataset_str)

            

        