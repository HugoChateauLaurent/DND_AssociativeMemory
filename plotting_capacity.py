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
fig_folder = "figures_exact"

def parse_sname_for_title(sname):
    if "tiny_" in sname or "imagenet_" in sname:
        return "ImageNet"
    if "mnist_" in sname:
        return "MNIST"
    else:
        return "CIFAR10"
    
def plot_separation_function_graph(Ns, imgs, beta,sep_fns, labels, image_perturb_fn = halve_continuous_img,sigma=1,f=manhattan_distance,use_norm = True,sep_param = 1, plot_results = False):
    corrects_list = [[] for i in range(len(sep_fns))]
    for i,(sep_fn, label) in enumerate(zip(sep_fns, labels)):
        for N in Ns:
            N_correct = PC_retrieve_store_continuous(imgs,N,beta=beta,num_plot=0,image_perturb_fn = image_perturb_fn,sigma = sigma,sep_fn = sep_fn,f=f,use_norm = use_norm,sep_param = sep_param)
            corrects_list[i].append(N_correct)

    if plot_results:
        plt.title("Memory Capacity by separation function")
        for i in range(len(sep_fns)):
            plt.plot(Ns, corrects_list[i], label = labels[i])

        plt.xlabel("Images Stored")
        plt.ylabel("Fraction Correctly Retrieved")
        plt.legend()
        plt.close()
    return np.array(corrects_list).reshape(len(sep_fns), len(Ns))
    

def N_runs_separation_function_graphs(N_runs, Ns, imgs, beta,sep_fns,fn_labels, fn_colors=None, fn_linestyles=None, f = euclidean_distance, sep_fn = separation_max, sep_param = 1, load_data = False,sname = "tiny_N_runs_separation_function_results_2.npy", figname = "tiny_N_runs_separation_functions_2.pdf", plot_results = True):
    if not load_data:
        N_corrects = []
        max_N = Ns[-1]
        for n in range(N_runs):
            X = imgs[(max_N*n):(max_N * (n+1))]
            corrects_list = plot_separation_function_graph(Ns, X, beta, sep_fns, fn_labels, f=f,sep_param=sep_param)
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
        # plt.title(dataset + " - " + f.__name__)
        xs = np.arange(0, len(Ns))
        for i in range(len(sep_fns)):
            if "Softmax" not in fn_labels or i!=fn_labels.index("Softmax"):
                if fn_colors is not None and fn_linestyles is not None:
                    plt.plot(Ns, mean_corrects[i,:],label=fn_labels[i], color=fn_colors[i], linestyle=fn_linestyles[i], zorder=-i)
                else:
                    plt.plot(Ns, mean_corrects[i,:],label=fn_labels[i], zorder=-i)
                plt.fill_between(Ns, mean_corrects[i,:] - std_corrects[i,:], mean_corrects[i,:]+std_corrects[i,:],alpha=ALPHA, zorder=-i)
        fig, ax = plt.gcf(), plt.gca()
        ax.spines[['right', 'top']].set_visible(False)
        plt.xlabel("Images Stored")
        plt.ylabel("Fraction Correctly Retrieved")
        plt.ylim(bottom=0,top=1)
        plt.legend()
        fig.tight_layout()
        plt.savefig(fig_folder+"/"+figname, format=SAVE_FORMAT)
        plt.close()
    return N_corrects

def run_separation_function_experiments(imgs, dataset_str, similarity_f):
    
    sep_fns = [separation_max, separation_2max, separation_5max, separation_10max, separation_50max, separation_identity]
    sep_labels = ["Max", "2-Max", "5-Max","10-Max", "50-Max", "Identity"]
    
    #Ns = [2,3,5,7,10,20,50,100,200,300]
    Ns = [5,7,10,20,50,100,200,300,500,700,1000]
    # Ns = [10,300]
    beta = 1
    N_runs = 10
    N_runs_separation_function_graphs(N_runs, Ns, imgs,  beta, sep_fns, sep_labels, f=similarity_f, load_data = LOAD_DATA, plot_results = PLOT_RESULTS,sname=similarity_f.__name__+"_"+dataset_str + "N_runs_separation_function_results.npy", figname=similarity_f.__name__+"_"+dataset_str+ "N_runs_separation_functions." + SAVE_FORMAT)

def run_separation_function_experiments_betas(imgs, dataset_str, similarity_f):
    
    sep_softmaxs = [separation_softmax_beta10000, separation_softmax_beta1000, separation_softmax_beta100, separation_softmax_beta10, separation_softmax]
    softmaxs_labels = [r"$\beta=10000$", r"$\beta=1000$", r"$\beta=100$", r"$\beta=10$", r"$\beta=1$"]
    
    #Ns = [2,3,5,7,10,20,50,100,200,300]
    Ns = [5,7,10,20,50,100,200,300,500,700,1000]
    # Ns = [10,300]
    beta = 1
    N_runs = 10
    N_runs_separation_function_graphs(N_runs, Ns, imgs,  beta, sep_softmaxs, softmaxs_labels, f=similarity_f, load_data = LOAD_DATA, plot_results = PLOT_RESULTS,sname=similarity_f.__name__+"_"+dataset_str + "N_runs_separation_function_betas_results.npy", figname=similarity_f.__name__+"_"+dataset_str+ "N_runs_separation_functions_betas." + SAVE_FORMAT)


    
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

            LOAD_DATA = False
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


            run_separation_function_experiments(imgs, dataset_str, similarity_f)
            # run_separation_function_experiments_betas(imgs, dataset_str, similarity_f)
            

        