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
sns.set_theme(style="darkgrid")


def draw(file_paths,dir,name):
    scoremarkers=["v","s","*","o","x","+"]
    # accmarkers=["v","s","*","o","x"]
    for i, path in enumerate(file_paths):
        fmri=pd.read_csv(path,sep=',',header=None,names=["score"],index_col=False)
        num=sum(1 for line in open(path))
        # fmri["score"]=fmri["score"]*100
        ax=sns.lineplot(x=range(1,num+1),y="score",err_style = "band",ci="sd",marker=scoremarkers[i],linewidth=3,
            # hue="region", style="event",
            data=fmri)
        plt.xlabel("index",fontsize=20)
        plt.ylabel('singular values',fontsize=20)
        # plt.yticks(np.arange(0, 60, 10))
        plt.legend([r"$\alpha=0.5$",r"$\alpha=1$", r"$\alpha=2$",r"$\alpha=3$",r"$\alpha=10$",r'$Adaptive$'],loc="upper right",fontsize=12)
        plt.savefig(os.path.join(dir+ name), format="pdf",bbox_inches="tight",dpi = 400)
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