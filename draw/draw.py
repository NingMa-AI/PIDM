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


def draw(file_paths,dir,draw="acc"):
    scoremarkers=["v","o","x","s"]
    accmarkers=["v","o","x","s"]

    if draw in "test_acc":

        for i, path in enumerate(file_paths):

            fmri=pd.read_csv(path)
            fmri["test_acc"]=fmri["test_acc"]*100
            ax=sns.lineplot(x="epoch",y="test_acc",err_style = "band",ci="sd", marker=accmarkers[i], linewidth=3,
                    # hue="region", style="event",
                    data=fmri)
        plt.xlabel("epoch",fontsize=20)
        plt.ylabel('Accuracies, (%)',fontsize=20)
        plt.yticks(np.arange(35, 65, 5))

        plt.legend(["EM", r'+BNM',"+Div","+LP"],loc="lower left",fontsize=15)
        plt.savefig(os.path.join(dir+ "show_ACC.pdf"), format="pdf",bbox_inches="tight",dpi = 400)
       
    else:
        for i, path in enumerate(file_paths):

            fmri=pd.read_csv(path)

            ax=sns.lineplot(x="epoch",y="score",err_style = "band",ci="sd",marker=scoremarkers[i],linewidth=3,
                # hue="region", style="event",
                data=fmri)
            plt.ylabel('E1 scores',fontsize=20)
            plt.yticks(np.arange(0, 0.6, 0.1))
            plt.legend(["EM", r'+BNM',"+Div","+LP"],loc="upper left",fontsize=15)
            plt.savefig(os.path.join(dir+ "show_SCORE.pdf"), format="pdf",bbox_inches="tight",dpi = 400)

    
    # plt.show()

def draw_single(file_path):
    fmri=pd.read_csv(file_path)
    fmri["test_acc"]=fmri["test_acc"]*100
    fmri["zero"]=fmri["zero"]*100
    fmri["score"]=fmri["score"]*100
    fmri["two"]=fmri["two"]*100

    ax=sns.lineplot(x="epoch",y="test_acc",err_style = "band",ci="sd", marker="v", linewidth=3, color="silver",
             # hue="region", style="event",
             data=fmri)

    ax=sns.lineplot(x="epoch",y="zero",err_style = "band",ci="sd",marker="o",linewidth=3,
             # hue="region", style="event",
             data=fmri)

    ax=sns.lineplot(x="epoch",y="score",err_style = "band",ci="sd",marker="s",linewidth=3,
             # hue="region", style="event",
             data=fmri)
    
    ax=sns.lineplot(x="epoch",y="two",err_style = "band",ci="sd",marker="*",linewidth=3,
             # hue="region", style="event",
             data=fmri)
    # ax=sns.lineplot(x="epoch",y="entropy",err_style = "band",ci="sd",marker="s",linewidth=3,
    #         # hue="region", style="event",
    #         data=fmri)

    plt.xlabel("epochs",fontsize=20)
    plt.ylabel('percentages',fontsize=20)
    plt.yticks(np.arange(0, 100, 10))
    plt.legend(["Acc", r'$S_{\mu=0}$' , r'$S_{\mu=1}$', r'$S_{\mu>1}$'],loc="upper right",fontsize=12)
    plt.savefig(os.path.join(file_path + "show.pdf"), format="pdf",bbox_inches="tight",dpi = 400)
    # plt.show()
    # print("done")
    # plt.show()

# draw(["/data/maning/git/shot/ssda/2021_07_10office-home/seed2021/office-home/Real_World_norm0_temp0.05_lr0.001/tb_Real_World2Clipart_lr0.001_unl_ent0.3_unl_w0.0_vat_w0_div_w0.0bnm0.0_num1.csv",
# "/data/maning/git/shot/ssda/2021_07_07office-home/seed2021/office-home/Real_World_norm0_temp0.05_lr0.001/tb_Real_World2Clipart_lr0.001_unl_ent0.1_unl_w0.0_vat_w0_div_w0.0bnm1.0_num1.csv",

# "/data/maning/git/shot/ssda/2021_07_10office-home/seed2021/office-home/Real_World_norm0_temp0.05_lr0.001/tb_Real_World2Clipart_lr0.001_unl_ent0.3_unl_w0.0_vat_w0_div_w1.0bnm0.0_num1.csv",

# "/data/maning/git/shot/ssda/2021_07_10office-home/seed2021/office-home/Real_World_norm0_temp0.05_lr0.001/tb_Real_World2Clipart_lr0.001_unl_ent0.1_unl_w0.1_vat_w0_div_w0.0bnm0.0_num1.csv"

# ],"/data/maning/git/shot/ssda/2021_07_10office-home/seed2021/office-home/Real_World_norm0_temp0.05_lr0.001/", "test_acc")

#em
# draw_single("/data/maning/git/shot/ssda/2021_02_05office-home/seed2021/Office-31/webcam_norm1_temp0.05_lr0.001/tb_webcam2amazon_lr0.001_unl_ent0.2_unl_w0.0_vat_w0_div_w0bnm0_num1.csv")

#bnm
# draw_single("/data/maning/git/shot/ssda/2021_02_05office-home/seed2021/Office-31/webcam_norm1_temp0.05_lr0.001/tb_webcam2amazon_lr0.001_unl_ent0.3_unl_w0.0_vat_w0_div_w0bnm1.0_num1.csv")

#div
# draw_single("/data/maning/git/shot/ssda/2021_02_05office-home/seed2021/Office-31/webcam_norm1_temp0.05_lr0.001/tb_webcam2amazon_lr0.001_unl_ent1.0_unl_w0.0_vat_w0_div_w1.0bnm0_num1.csv")

#lp
# draw_single("/data/maning/git/shot/ssda/2021_02_05office-home/seed2021/Office-31/webcam_norm1_temp0.05_lr0.001/tb_webcam2amazon_lr0.001_unl_ent0.3_unl_w0.3_vat_w0_div_w0bnm0_num1.csv")