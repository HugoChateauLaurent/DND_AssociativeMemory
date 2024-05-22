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
K2BETA = 500

fig_folder = "figures_exact"

def parse_sname_for_title(sname):
    if "tiny_" in sname or "imagenet_" in sname:
        return "ImageNet"
    if "mnist_" in sname:
        return "MNIST"
    else:
        return "CIFAR10"
    
def plot_mask_frac_graphs(N, imgs, beta, fs, labels, mask_fracs, similarity_f,use_norm = True, image_perturb_fn = mask_continuous_img):
    corrects_list = [[] for i in range(len(mask_fracs))]
    for i,mask_frac in enumerate(mask_fracs):
        corrects = [[] for i in range(len(fs))]
        for j, (f,label) in enumerate(zip(fs,labels)):
            N_correct = PC_retrieve_store_continuous(imgs, N, beta=beta, num_plot=0,f=similarity_f,sigma=mask_frac,image_perturb_fn=image_perturb_fn,use_norm = use_norm,sep_fn = f)
            corrects[j].append(deepcopy(N_correct))
        corrects_list[i].append(np.array(corrects))
    corrects_list = np.array(corrects_list)
    return corrects_list.reshape(len(mask_fracs),len(fs))

def N_runs_mask_frac_graphs(N_runs, N, imgs, beta,fs,fn_labels, mask_fracs, similarity_f, load_data = False,sname = "tiny_N_mask_frac_results.npy", figname = "tiny_N_runs_mask_fracs.jpg", plot_results = True, image_perturb_fn= mask_continuous_img):
    if not load_data:
        N_corrects = []
        for n in range(N_runs):
            X = imgs[(N*n):(N * (n+1))]
            corrects_list = plot_mask_frac_graphs(N, X, beta, fs, fn_labels, mask_fracs, similarity_f = similarity_f,image_perturb_fn= image_perturb_fn)
            N_corrects.append(corrects_list)
        N_corrects = np.array(N_corrects)
        np.save(fig_folder+"/"+sname, N_corrects)
    else:
        N_corrects = np.load(fig_folder+"/"+sname)

    print(N_corrects.shape)

    for i in range(len(fn_labels)):
        print(fn_labels[i], np.mean(N_corrects[:,i,:]), "+-", np.std(N_corrects[:,i,:]))
    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        # begin plot
        # dataset = parse_sname_for_title(sname)
        # plt.title(dataset + " Fraction Masked")
        for i in range(len(fs)):
            if "Softmax" not in fn_labels or i!=fn_labels.index("Softmax"):
                plt.plot(mask_fracs, mean_corrects[:,i],label=fn_labels[i])
                plt.fill_between(mask_fracs, mean_corrects[:,i] - std_corrects[:,i], mean_corrects[:,i]+std_corrects[:,i],alpha=0.5)
        fig, ax = plt.gcf(), plt.gca()
        ax.spines[['right', 'top']].set_visible(False)
        plt.xlabel("Fraction Masked")
        plt.ylabel("Fraction Correctly Retrieved")
        plt.ylim(bottom=0,top=1)
        plt.legend()
        fig.tight_layout()
        plt.savefig(fig_folder+"/"+figname, format=SAVE_FORMAT)
        plt.close()
    return N_corrects

def plot_noise_level_graphs(N, imgs, beta, fs, labels, sigmas, similarity_f,use_norm = True):
    corrects_list = [[] for i in range(len(sigmas))]
    for i,sigma in enumerate(sigmas):
        corrects = [[] for i in range(len(fs))]
        for j, (f,label) in enumerate(zip(fs,labels)):
            N_correct = PC_retrieve_store_continuous(imgs, N, beta=beta, num_plot=0,f=similarity_f,sigma=sigma,image_perturb_fn=gaussian_perturb_image,use_norm = use_norm,sep_fn = f)
            corrects[j].append(deepcopy(N_correct))
        corrects_list[i].append(np.array(corrects))
    corrects_list = np.array(corrects_list)
    return corrects_list.reshape(len(sigmas), len(fs))

def N_runs_noise_level_graphs(N_runs, N, imgs, beta,fs,fn_labels, sigmas, similarity_f, load_data = False,sname = "tiny_N_noise_level_results.npy", figname = "tiny_N_runs_noise_levels.jpg", plot_results = True):
    if not load_data:
        N_corrects = []
        for n in range(N_runs):
            X = imgs[(N*n):(N * (n+1))]
            corrects_list = plot_noise_level_graphs(N, X, beta, fs, fn_labels, sigmas, similarity_f = similarity_f)
            N_corrects.append(corrects_list)
        N_corrects = np.array(N_corrects)
        np.save(fig_folder+"/"+sname, N_corrects)
    else:
        N_corrects = np.load(fig_folder+"/"+sname)

    for i in range(len(fn_labels)):
        print(fn_labels[i], np.mean(N_corrects[:,:,i]), "+-", np.std(N_corrects[:,:,i]))
    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        # begin plot
        # dataset = parse_sname_for_title(sname)
        # plt.title(dataset + " Noise Levels")
        for i in range(len(fs)):
            if "Softmax" not in fn_labels or i!=fn_labels.index("Softmax"):
                plt.plot(sigmas, mean_corrects[:,i],label=fn_labels[i])
                plt.fill_between(sigmas, mean_corrects[:,i] - std_corrects[:,i], mean_corrects[:,i]+std_corrects[:,i],alpha=0.5)
        
        fig, ax = plt.gcf(), plt.gca()
        ax.spines[['right', 'top']].set_visible(False)
        plt.xlabel("Noise variance (sigma)")
        plt.ylabel("Fraction Correctly Retrieved")
        plt.ylim(bottom=0,top=1)
        plt.legend()
        fig.tight_layout()
        plt.savefig(fig_folder+"/"+figname, format=SAVE_FORMAT)
        plt.close()
    return N_corrects

def plot_noise_level_graphs_beta_vs_k(N, imgs, beta, f, params, sigma, similarity_f,use_norm = True):
    corrects_list = []
    for i,param in enumerate(params):
        N_correct = PC_retrieve_store_continuous(imgs, N, beta=beta, sep_param=param, num_plot=0,f=similarity_f,sigma=sigma,image_perturb_fn=gaussian_perturb_image,use_norm = use_norm,sep_fn = f)
        corrects_list.append(deepcopy(N_correct))
    corrects_list = np.array(corrects_list)
    return corrects_list.reshape(len(params))

def N_runs_noise_level_graphs_beta_vs_k(N_runs, N, imgs, betas, ks, sigma, similarity_f, beta=1, load_data = False,sname = "tiny_N_noise_level_results.npy", figname = "tiny_N_runs_noise_levels.jpg", plot_results = True):
    
    if not load_data:
        N_corrects_betas, N_corrects_ks = [], []
        for n in range(N_runs):
            X = imgs[(N*n):(N * (n+1))]
            corrects_list_betas = plot_noise_level_graphs_beta_vs_k(N, X, beta, separation_softmax, betas, sigma, similarity_f = similarity_f)
            corrects_list_ks = plot_noise_level_graphs_beta_vs_k(N, X, beta, separation_kmax, ks, sigma, similarity_f = similarity_f)
            N_corrects_betas.append(corrects_list_betas)
            N_corrects_ks.append(corrects_list_ks)
        N_corrects_betas = np.array(N_corrects_betas)
        N_corrects_ks = np.array(N_corrects_ks)
        np.save(fig_folder+"/"+sname, [N_corrects_betas, N_corrects_ks])
    else:
        N_corrects_betas, N_corrects_ks = np.load(fig_folder+"/"+sname)

    print(N_corrects_betas.shape, N_corrects_ks.shape)

    

    
    if plot_results:

        fig, ax = plt.subplots(constrained_layout=True)
        # secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))

        mean_corrects_betas = np.mean(N_corrects_betas,axis=0)
        std_corrects_betas = np.std(N_corrects_betas, axis=0)
        mean_corrects_ks = np.mean(N_corrects_ks,axis=0)
        std_corrects_ks = np.std(N_corrects_ks, axis=0)
        # begin plot
        # dataset = parse_sname_for_title(sname)
        # plt.title(dataset + " Noise Levels")
        plt.plot(ks, mean_corrects_betas[::-1],label="Softmax")
        plt.fill_between(ks, mean_corrects_betas[::-1] - std_corrects_betas[::-1], mean_corrects_betas[::-1] + std_corrects_betas[::-1],alpha=0.5)
        plt.plot(ks, mean_corrects_ks,label="k-Max")
        plt.fill_between(ks, mean_corrects_ks - std_corrects_ks, mean_corrects_ks + std_corrects_ks,alpha=0.5)
        
        fig, ax = plt.gcf(), plt.gca()
        ax.spines[['right', 'top']].set_visible(False)
        plt.xlabel(r"$k$")
        # plt.xlabel(r"$k$" "\n" r"$\beta/{K2BETA}$".format(K2BETA=K2BETA))
        plt.ylabel("Fraction Correctly Retrieved")
        # plt.ylim(bottom=0,top=1)

        
        def k2beta_abstract(x):
            return x * K2BETA
        def beta2k_abstract(x):
            return x / K2BETA
        def k2beta(x):
            return k2beta_abstract(ks[-1]) - k2beta_abstract(x) + K2BETA
        def beta2k(x):
            return beta2k_abstract(k2beta_abstract(ks[-1]) - x - K2BETA)
        
        secax = ax.secondary_xaxis('top', functions=(k2beta, beta2k))
        secax.set_xlabel(r"$\beta$")

        print("Best beta", betas[mean_corrects_betas.argmax()], mean_corrects_betas.max(), "+-", std_corrects_betas[mean_corrects_betas.argmax()])
        print("Best k", ks[mean_corrects_ks.argmax()], mean_corrects_ks.max(), "+-", std_corrects_ks[mean_corrects_ks.argmax()])
        plt.legend()
        fig.tight_layout()
        plt.savefig(fig_folder+"/"+figname, format=SAVE_FORMAT)
        plt.close()

        
    return N_corrects_ks, N_corrects_betas

def run_noise_levels_experiments(imgs, dataset_str, similarity_f):
    sigmas = [0.05,0.1,0.2,0.3,0.5,0.8,1,1.5]#,2]
    N = 100
    N_runs = 10
    sep_fns = [separation_max, separation_2max, separation_5max, separation_10max, separation_50max, separation_identity]
    sep_labels = ["Max", "2-Max", "5-Max","10-Max", "50-Max", "Identity"]
    beta = 1
    corrects_list = N_runs_noise_level_graphs(N_runs, N,imgs,beta,sep_fns,sep_labels,sigmas, similarity_f=similarity_f, load_data=LOAD_DATA,plot_results = PLOT_RESULTS,sname=similarity_f.__name__+"_"+dataset_str + "2_N_noise_level_results.npy", figname = similarity_f.__name__+"_"+dataset_str + "N_runs_noise_levels." + SAVE_FORMAT)

def run_noise_levels_experiments_betas(imgs, dataset_str, similarity_f):
    sigmas = [0.05,0.1,0.2,0.3,0.5,0.8,1,1.5]#,2]
    N = 100
    N_runs = 10
    beta = 1

    sep_softmaxs = [separation_softmax_beta10000, separation_softmax_beta1000, separation_softmax_beta100, separation_softmax_beta10, separation_softmax]
    softmaxs_labels = [r"$\beta=10000$", r"$\beta=1000$", r"$\beta=100$", r"$\beta=10$", r"$\beta=1$"]
    
    corrects_list = N_runs_noise_level_graphs(N_runs, N,imgs,beta,sep_softmaxs,softmaxs_labels,sigmas, similarity_f=similarity_f, load_data=LOAD_DATA,plot_results = PLOT_RESULTS,sname=similarity_f.__name__+"_"+dataset_str + "2_N_noise_level_betas_results.npy", figname = similarity_f.__name__+"_"+dataset_str + "N_runs_noise_levels_betas." + SAVE_FORMAT)

def run_noise_levels_experiments_beta_vs_k(imgs, dataset_str, similarity_f):
    ks = np.arange(1,101,step=1)#11, )
    betas = ks * K2BETA

    assert len(ks)==len(betas)

    N = 100
    N_runs = 10
    sigma = {
        "mnist_": 1., 
        "cifar_10_":.75, 
        "tiny_":.75,
    }[dataset_str]
    
    corrects_list = N_runs_noise_level_graphs_beta_vs_k(N_runs, N,imgs,betas,ks,sigma, similarity_f=similarity_f, load_data=LOAD_DATA,plot_results = PLOT_RESULTS,sname=similarity_f.__name__+"_"+dataset_str + "2_N_noise_level_beta_vs_k_results.npy", figname = similarity_f.__name__+"_"+dataset_str + "N_runs_noise_levels_beta_vs_k." + SAVE_FORMAT)


def run_frac_masking_experiments(imgs, dataset_str, similarity_f):
    mask_fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    N = 50
    N_runs =5
    sep_fns = [separation_max, separation_2max, separation_5max, separation_10max, separation_50max, separation_softmax, separation_identity]
    sep_labels = ["Max", "2-Max", "5-Max","10-Max", "50-Max", "Softmax", "Identity"]
    beta = 1
    mask_frac_corrects = N_runs_mask_frac_graphs(N_runs,N,imgs,beta,sep_fns,sep_labels,mask_fracs,similarity_f=similarity_f,load_data = LOAD_DATA,plot_results=PLOT_RESULTS,sname = similarity_f.__name__+"_"+dataset_str + "N_mask_frac_results.npy", figname = similarity_f.__name__+"_"+dataset_str + "N_runs_mask_fracs." + SAVE_FORMAT)
    
if __name__ == '__main__':

    #dataset_str = "mnist_longer_capacity_"
    dataset_str = "cifar_10_"
    #dataset_str = ""
    dataset_str = "mnist_"
    dataset_str = "tiny_"
    #separation functions
    #dataset_str = "imagenet_"

    for similarity_f in [euclidean_distance, manhattan_distance]:
        print(similarity_f.__name__)

        for dataset_str in ["mnist_", "cifar_10_", "tiny_"]:

            print(dataset_str)

            LOAD_DATA = True
            PLOT_RESULTS = True
            imgs = []
            if not LOAD_DATA:
                if dataset_str == "tiny_" or dataset_str == "imagenet_" or dataset_str == "tiny_2_":
                    imgs = load_tiny_imagenet(N_imgs=10000)
                if dataset_str == "mnist_" or dataset_str == "mnist_longer_capacity_":
                    trainset_mnist, testset_mnist = load_mnist(60000)
                    imgs = trainset_mnist[0][0]
                if dataset_str == "" or dataset_str == "cifar_10_" or dataset_str == "cifar":
                    trainset_cifar, testset_cifar = get_cifar10(10000)
                    imgs = trainset_cifar[0][0]


            # print("\n Noise levels betas")
            # run_noise_levels_experiments_betas(imgs, dataset_str, similarity_f)
            print("\n Noise levels beta vs k")
            run_noise_levels_experiments_beta_vs_k(imgs, dataset_str, similarity_f)
            # noise levels (Fig. 4 Top)
            # print("\n Noise levels")
            # run_noise_levels_experiments(imgs, dataset_str, similarity_f)
            # print("\n Frac masking")
            # # frac masking (Fig. 4 Bottom)
            # run_frac_masking_experiments(imgs, dataset_str, similarity_f)
            

        