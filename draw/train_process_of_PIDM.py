"""
draw singular values of DomainNet with resnet34
"""
import csv,os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
import matplotlib
import numpy as np
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
# sns.set_theme(style="darkgrid")

sns.set_theme(style="whitegrid", palette="pastel")


def draw(file_paths,dir,name):
    scoremarkers=["v","s","*","o","x","+"]
    # accmarkers=["v","s","*","o","x"]
    for i, path in enumerate(file_paths):
        fmri=pd.read_csv(path,sep=',',) #header=None,names=["score"],index_col=False
        num=sum(1 for line in open(path))
        # fmri["score"]=fmri["score"]*100
        # sns.barplot(x="alpha", y="RS", data=fmri)
        # sns.barplot(x="alpha", y="CS", data=tips)
        # sns.barplot(x="alpha", y="SP", data=tips)
        fmri["train_acc"]=fmri["train_acc"]*100
        fmri["val_acc"]=fmri["val_acc"]*100
        fmri["test_acc"]=fmri["test_acc"]*100
        a=0
        ax=sns.lineplot(x="epoch",y="train_acc",err_style = "band",ci="sd",marker=scoremarkers[a],linewidth=3,
            # hue="region", style="event",
            data=fmri)
        a=a+1
        ax=sns.lineplot(x="epoch",y="val_acc",err_style = "band",ci="sd",marker=scoremarkers[a],linewidth=3,
            # hue="region", style="event",
            data=fmri)
        a=a+1
        ax=sns.lineplot(x="epoch",y="test_acc",err_style = "band",ci="sd",marker=scoremarkers[a],linewidth=3,
            # hue="region", style="event",
            data=fmri)
        plt.xlabel("epoch",fontsize=20)
        plt.ylabel('Accuracy,%',fontsize=20)
        plt.yticks(np.arange(55, 100, 5))
        plt.legend([r"Training",r"Validation", r"Test"],loc="center right",fontsize=12)
        plt.savefig(os.path.join(dir+ name), format="pdf",bbox_inches="tight",dpi = 400)
        # plt.clf()

draw(["/data/maning/git/shot/draw/training-process.csv"
],
"/data/maning/git/shot/draw/", "training-process.pdf")
        # plt.clf()

# R-C  (domainnet)
# draw(["/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2clipart_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha0.5_num3_clipart.csv",
#     "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2clipart_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha1.0_num3_clipart.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2clipart_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha2.0_num3_clipart.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2clipart_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha3.0_num3_clipart.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2clipart_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha10.0_num3_clipart.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2clipart_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear1_wnl0_alpha0.75_num3_clipart.csv"
# ],
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/", "real2clipart_alphas1.pdf")

#R-S
# draw(["/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha0.5_num3_sketch.csv",
#     "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha1.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha2.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha3.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha10.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/tar_real2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear1_wnl0_alpha0.75_num3_sketch.csv"
# ],
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/real_norm1_temp0.05_lr0.001/","real2sketch_alphas1.pdf")

# C-S
# draw(["/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/clipart_norm1_temp0.05_lr0.001/tar_clipart2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha0.5_num3_sketch.csv",
#     "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/clipart_norm1_temp0.05_lr0.001/tar_clipart2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha1.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/clipart_norm1_temp0.05_lr0.001/tar_clipart2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha2.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/clipart_norm1_temp0.05_lr0.001/tar_clipart2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha3.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/clipart_norm1_temp0.05_lr0.001/tar_clipart2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha10.0_num3_sketch.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/clipart_norm1_temp0.05_lr0.001/tar_clipart2sketch_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear1_wnl0_alpha0.75_num3_sketch.csv"
# ],
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/clipart_norm1_temp0.05_lr0.001/","clipart2sketch_alphas1.pdf")


# S-P
# draw([
#     "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/sketch_norm1_temp0.05_lr0.001/tar_sketch2painting_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha0.5_num3_painting.csv",
#     "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/sketch_norm1_temp0.05_lr0.001/tar_sketch2painting_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha1.0_num3_painting.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/sketch_norm1_temp0.05_lr0.001/tar_sketch2painting_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha2.0_num3_painting.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/sketch_norm1_temp0.05_lr0.001/tar_sketch2painting_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha3.0_num3_painting.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/sketch_norm1_temp0.05_lr0.001/tar_sketch2painting_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear0_wnl0_alpha10.0_num3_painting.csv",
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/sketch_norm1_temp0.05_lr0.001/tar_sketch2painting_lr0.001MNPC5_im1.0_u200.0_unlent0.0_nonlinear1_wnl0_alpha0.75_num3_painting.csv"
# ],
# "/data/maning/git/shot/ssda/2022_05_10multi/seed2022/multi/sketch_norm1_temp0.05_lr0.001/","sketch2painting_alphas1.pdf")