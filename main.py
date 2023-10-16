from utils import *
import argparse
import os
import os.path as osp
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from data_pro.mixmatch_return_dataset import return_dataset,return_dataloader_by_UPS,return_dataloader_by_progressive_UPS
from models.SSDA_basenet import *
from models.SSDA_resnet import *
from scipy.spatial.distance import cdist
from copy import deepcopy
import contextlib
# from loss_utils import *
import scipy
import scipy.stats

from itertools import cycle

def train_source(args):
    if os.path.exists(osp.join(args.output_dir, "source_C_val.pt")):
        print("train_file exist,",args.output_dir)
        return 0
    source_loader,source_val_loader, _, _, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
    netF,netC,_=get_model(args)
    netF=netF.cuda()
    netC=netC.cuda()
    param_group = []
    learning_rate = args.lr
    for k, v in netF.features.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate}]

    for k, v in netF.bottle_neck.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate * 10}]

    for k, v in netC.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate * 10}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optimizer = optim.Adam(param_group, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=0.5, patience=50,
                                                                verbose=True, min_lr=1e-6)
    # scaler = torch.cuda.amp.GradScaler()
    acc_init = 0
    for epoch in (range(args.max_epoch_source)):
        netF.train()
        netC.train()
        # loss=nn.CrossEntropyLoss().cuda()
        total_losses,recon_losses,classifier_losses=[],[],[]
        iter_source = iter(source_loader)
        for _, (inputs_source, labels_source) in tqdm(enumerate(iter_source), leave=False):
            if inputs_source.size(0) == 1:
                continue
            # inputs_source, labels_source = inputs_source, labels_source
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            # print(inputs_source.shape)
            # print("input:",inputs_source.shape,labels_source)
            # with torch.cuda.amp.autocast():
            embeddings=netF(inputs_source)

            outputs_source = netC(embeddings)
            # logits,margin_logits = netC(embeddings,labels_source)

            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source,
                                                                                                labels_source,T=1)
            # classifier_loss = loss(outputs_source,labels_source)
            total_loss=classifier_loss

            # print("loss",total_loss)
            total_losses.append(total_loss.item())
            classifier_losses.append(classifier_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # break
        # print("total_loss: {:.6f}, classify loss: {:.6f},recon_loss:{:.6f}".format())
        netF.eval()
        netC.eval()

        scheduler.step(np.mean(total_losses))
        acc_s_tr, _ = cal_acc(source_loader, netF, netC)
        acc_s_te, _ = cal_acc(source_val_loader, netF, netC)
        log_str = 'train_source , Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%, total_loss: {:.6f}, classify loss: {:.6f},'\
            .format(args.s+"2"+args.t, epoch + 1, args.max_epoch_source, acc_s_tr * 100, acc_s_te * 100,np.mean(total_losses),np.mean(classifier_losses))
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')

        if acc_s_te >= acc_init:
            acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
            torch.save(best_netF, osp.join(args.output_dir, "source_F_val.pt"))
            torch.save(best_netC, osp.join(args.output_dir, "source_C_val.pt"))
    #
    # torch.save(best_netF, osp.join(args.output_dir, "source_F_val.pt"))
    # torch.save(best_netC, osp.join(args.output_dir, "source_C_val.pt"))

    return netF, netC


def test_target(args):
    _, _,_, _, _, target_loader_test, class_list = return_dataset(args)
    netF, netC, _ = get_model(args)


    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))

    netF=nn.DataParallel(netF,device_ids=[0])
    netC=nn.DataParallel(netC,device_ids=[0])

    netC=netC.cuda()
    netF=netF.cuda()
    netF.eval()
    netC.eval()

    acc, _,= cal_acc(target_loader_test, netF, netC)
    log_str = 'test_target Task: {}, Accuracy = {:.2f}%'.format(args.s+"2"+args.t, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    # file_name1=osp.join(args.output_dir, 'tar_' + args.s+"2"+args.t+"_lr"+str(args.lr)
    #                                     + "MNPC"+str(args.max_num_per_class)+ "_im"+str(args.im)
    #                                     + "_u"+str(args.lambda_u)+"_unlent"+str(args.unlent)+"_num"+str(args.num)+'embeding_pretrain.tsv')
    # file_name2 = osp.join(args.output_dir, 'tar_' + args.s + "2" + args.t + "_lr" + str(args.lr)
    #                         + "MNPC" + str(args.max_num_per_class) + "_im" + str(args.im)
    #                         + "_u" + str(args.lambda_u) + "_unlent" + str(args.unlent) + "_num" + str(
    #     args.num) + 'meta_pretrain.tsv')
    # np.savetxt(file_name1, emb,delimiter='\t')
    # np.savetxt(file_name2, label, delimiter='\t')


def test_target_svd(args):
    source_loader, _,_, _, _, target_loader_test, class_list = return_dataset(args)
    netF, netC, _ = get_model(args)
    name='tar_' + args.s+"2"+args.t+"_lr"+str(args.lr)+ "MNPC"+str(args.max_num_per_class)+ "_im"+str(args.im)+ "_u"+str(args.lambda_u)+"_unlent"+str(args.unlent)+"_nonlinear"+str(args.nonlinear)+"_wnl"+str(args.wnl)+"_alpha"+str(args.alpha)+"_num"+str(args.num)+"_"

    args.modelpath = args.output_dir + '/{}target_F.pt'.format(name)
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/{}target_C.pt'.format(name)
    netC.load_state_dict(torch.load(args.modelpath))
    netC=netC.cuda()
    netF=netF.cuda()
    netF.eval()
    netC.eval()

    all_fea,all_output,all_label=compute(target_loader_test,netF,netC,args)
    len_data=len(all_fea)
    bs=args.batch_size
    target_s=[]
    target_u=[]
    for i in range(0,len_data-bs,bs):
        fea=all_fea[i:i+bs]
        u,s,v=torch.svd(fea.t())
        target_s.append(s.detach.cpu())
        target_u.append(u.detach.cpu())
    target_S= torch.mean(torch.cat(target_s,dim=0),dim=0)
    target_U= torch.mean(torch.cat(target_u,dim=0),dim=0)

    all_fea,all_output,all_label=compute(source_loader,netF,netC,args)
    len_data=len(all_fea)
    bs=args.batch_size
    source_s=[]
    source_u=[]
    for i in range(0,len_data-bs,bs):
        fea=all_fea[i:i+bs]
        u,s,v=torch.svd(fea.t())
        source_s.append(s.detach.cpu())
        source_u.append(u.detach.cpu())
    source_S= torch.mean(torch.cat(target_s,dim=0),dim=0)
    source_U= torch.mean(torch.cat(target_u,dim=0),dim=0)

    p_s, cospa, p_t = torch.svd(torch.mm(source_U.t(), target_U))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    subspace_distance=torch.norm(sinpa,1)
    source_S=source_S.numpy()
    target_S=target_S.numpy()
    print("subspace_distance: ", subspace_distance)
    np.savetxt(args.output_dir+"/"+name+".csv",source_S.reshpae(-1),delimiter=",")
    np.savetxt(args.output_dir+"/"+name+".csv",source_S.reshpae(-1),delimiter=",")










    





def train_target(args):
    source_loader, source_val_loader, target_loader, target_loader_unl, \
    target_loader_val, target_loader_test, (labeled_target, unlabeled_target)=return_dataset(args)

    # args.lr=0.01
    len_target_loader=len(target_loader)
    len_target_loader_unl=len(target_loader_unl)

    netF, netC, netD = get_model(args)
    # print(get_para_num(netF))
    # # print(get_para_num(netF.bottle_neck))
    # print(get_para_num(netC))
    # print(get_para_num(netD))

    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF = netF.cuda()
    netC = netC.cuda()
    param_group = []
    # for k, v in netF.named_parameters():
    #     v.requires_grad=True
    #     param_group += [{'params': v, 'lr': args.lr}]

    for k, v in netF.features.named_parameters():
        v.requires_grad=True
        param_group += [{'params': v, 'lr': args.lr}]

    for k, v in netF.bottle_neck.named_parameters():
        v.requires_grad=True
        param_group += [{'params': v, 'lr': args.lr*10}]

    if args.update_cls==1:
        for k, v in netC.named_parameters():
            v.requires_grad = True
            param_group += [{'params': v, 'lr': args.lr}]
    else:
        for k, v in netC.named_parameters():
            v.requires_grad = False
            # param_group += [{'params': v, 'lr': args.lr}]

    # for k, v in netD.named_parameters():
    #     v.requires_grad=True
    #     param_group += [{'params': v, 'lr': args.lr*10}]


    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.8, patience=20,
                                                            verbose=True, min_lr=1e-5)
    # vat_loss=VATLoss()

    scaler = torch.cuda.amp.GradScaler()
    netF=nn.DataParallel(netF,device_ids=[0])
    netC=nn.DataParallel(netC,device_ids=[0])

    max_pred_acc=-1
    best_test_acc = -1
    acc,acc_val=0,0
    best_F,bestC,bestD=None,None,None
    first_epoch_acc=-1

    psudo_acc=-1
    for epoch in (range(args.max_epoch_target)):
        if not args.max_num_per_class==0:
            args.cur_epoch=epoch+1
            # target_loader, target_loader_unl, psudo_acc= return_dataloader_by_UPS(args,netF,netC,target_loader_unl)
            target_loader, target_loader_unl, psudo_acc= return_dataloader_by_progressive_UPS(args,netF,netC,target_loader_unl)
            

        # psudo_acc=0
        # active_target_loader, active_target_loader_unl= target_loader, target_loader_unl

        #iter_target = iter(active_target_loader)
        #iter_target_unl = iter(active_target_loader_unl)

        total_losses, classifier_losses,unl_losses,ws,entropy_losses,div_losses,ls= [], [],[],[],[],[],[]

        netF.train()
        netC.train()

        #for step, ((inputs_x, targets_x), ((inputs_u,inputs_u2),_)) in tqdm(enumerate(zip(cycle(active_target_loader),active_target_loader_unl)), leave=False):
        for step, ((inputs_x, targets_x), ((inputs_u,inputs_u2),_)) in enumerate(zip(cycle(target_loader),target_loader_unl)):
        # for step, (unlabeled_target, _) in (enumerate(iter_target_unl)):
        #     if unlabeled_target.size(0) == 1:
        #         continue
            start=time.time()
            #if step % len_target_loader==0:
                #print("dataloader")
                #iter_target=iter(active_target_loader)
            # inputs_test = inputs_test
            #inputs_x,targets_x=next(iter_target)

            batch_size = inputs_x.size(0)

            # print(batch_size,inputs_x.shape,inputs_u.shape,inputs_u2.shape)
            # Transform label to one-hot
            targets_x = torch.zeros(batch_size, args.class_num).scatter_(1, targets_x.view(-1, 1).long(), 1)
            #print(time.time()-start)
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            #print(time.time()-start)
            entropy_a=None
            #with torch.cuda.amp.autocast():
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                size=inputs_u.shape[0]
                #outputs_u = netC(netF(inputs_u))
                #outputs_u2 = netC(netF(inputs_u2))
                input_cat=torch.cat([inputs_u, inputs_u2], dim=0)
                out_cat=netC(netF(input_cat))
                outputs_u=out_cat[0:size]
                outputs_u2=out_cat[size:]

                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2

                entropy_a = Entropy(p)

                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            #for self mixup
            # all_inputs = torch.cat([ inputs_u, inputs_u2], dim=0)
            # all_targets = torch.cat([targets_u, targets_u], dim=0)

            l = np.random.beta(args.alpha, args.alpha)

            l =  max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]


            if args.nonlinear==1: 
                with torch.no_grad():

                    # softmax_out = nn.Softmax(dim=1)(netC(netF(input_a)))
                    # # entropy_a = torch.mean((1-F.sigmoid(Entropy(softmax_out))),dim=0, keepdim=True)*0.1
                    # entropy_a = Entropy(softmax_out)
                    # entropy_a = (F.sigmoid(Entropy(softmax_out)))*1.5
                    # # entropy_a[0:args.batch_size]=0

                    # entropy_a=entropy_a.unsqueeze(1)
                    # print("a",entropy_a,entropy_a.shape,target_a.shape)

                    # softmax_out = nn.Softmax(dim=1)(netC(netF(input_b)))
                    # # entropy_b =torch.mean((1-F.sigmoid(Entropy(softmax_out))),dim=0,keepdim=True)*0.1
                    # entropy_b =(F.sigmoid(Entropy(softmax_out)))

                    # entropy_b=entropy_b.unsqueeze(1)

                    # all_entropy=torch.cat([entropy_a,entropy_b], dim=1)

                    # print(torch.mean(all_entropy),all_entropy.shape)
                    # nn.Softmax(dim=1)(all_entropy)[:,0]
                    # l=

                    alpha=np.exp(torch.mean(entropy_a).item())
                    # alpha2=(torch.mean(entropy_a).item())

                    l = np.random.beta(alpha, 0.5)

                    # l =  min(l, 1 - l)
                    
                    # l=l.unsqueeze(1)
                    # print(torch.mean(l),l.shape)

                    # print("b",entropy_b,entropy_b.shape,input_b.shape)

                # mixed_input = l * input_a + (1 - l) * input_b

                # # mixed_input = l.reshape(-1,1,1,1) * input_a + (1-l.reshape(-1,1,1,1)) * input_b
                # mixed_target = l*target_a + target_b*(1-l)

            # else:
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + target_b*(1-l)
            
            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            #mixed_input = interleave(mixed_input, batch_size)

            ls.append(l)
            #logits = [netC(netF(mixed_input[0]))]

            #for input in mixed_input[1:]:
                #logits.append(netC(netF(input)))

            logits=netC(netF(torch.cat(mixed_input,dim=0)))

            logits=list(torch.split(logits, batch_size))


            # put interleaved samples backd
            #logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            targets_x = mixed_target[:batch_size]
            targets_u = mixed_target[batch_size:]
            probs_u = torch.softmax(logits_u, dim=1)

            Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))
            #withted unlabeled loss 
            # wnl
            if args.wnl==1:
                with torch.no_grad():
                    prb_u = torch.softmax(logits_u, dim=1)
                    prb_u_entropy=((Entropy(prb_u))).reshape(-1,1)
                    # print("min",torch.min(targets_u_entropy),"max",torch.max(targets_u_entropy), "mean",torch.mean(targets_u_entropy))
                    prb_u_entropy=prb_u_entropy.repeat(1,targets_u.shape[1])

                Lu = torch.mean((((probs_u - targets_u) )** 2) * ((1/prb_u_entropy**2)))
            else :
                Lu = torch.mean((probs_u - targets_u) ** 2)
            # print(step / len_target_loader_unl)
            # if args.wnl==1: w=args.lambda_u
            # else : w = args.lambda_u * exp_rampup(epoch + step / len_target_loader_unl, args.max_epoch_target)
            
            w = args.lambda_u * exp_rampup(epoch + step / len_target_loader_unl, args.max_epoch_target)
            # w = args.lambda_u

            ws.append(w)
            classifier_losses.append(Lx.item())
            unl_losses.append(Lu.item())
            total_loss =  Lx + w * Lu


            # im_loss = 0
            # softmax_out = nn.Softmax(dim=1)(netC(netF(inputs_u)))
            # un_labeled_entropy = torch.mean(Entropy(softmax_out))
            # im_loss+=args.unlent*un_labeled_entropy
            # entropy_losses.append(un_labeled_entropy.item())
            # entropy_losses.append(0)

            # if args.nonlinear==1: 
            #     args.alpha=un_labeled_entropy.item()
    
            # msoftmax = softmax_out.mean(dim=0)
            # tmp = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            # div_losses.append(tmp.item())
            # im_loss -= args.div_w * tmp
            #
            # # im_loss= im_loss
            # # scaler.scale(im_loss).backward()
            # # im_loss.backward()

            # total_loss=total_loss+im_loss* args.im


            total_losses.append(total_loss.item())
            # total_loss

            #scaler.scale(total_loss).backward()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            #scaler.step(optimizer)
            #scaler.update()
            #optimizer.zero_grad()


        netF.eval()
        netC.eval()
        if epoch %5==0:
            acc, _ = cal_acc(target_loader_test, netF, netC)
            train_acc, _ = cal_acc(target_loader, netF, netC)
            acc_val, _ = cal_acc(target_loader_val, netF, netC)

            training_process={"epoch": epoch,"train_acc":train_acc, "val_acc":acc_val, "test_acc":acc}
            # training_process["train_acc"]=train_acc
            # training_process["train_acc"]=train_acc

            write_csv_file("training-process.csv",training_process)

            log_str = 'tra_tgt: {}, I:{}/{}; test_acc = {:.2f}% ,acc_val = {:.2f}% ,total_l: {:.5f}, Lx: {:.5f}, Lu: {:.8f}, w: {:.5f},ent_l: {:.3f}, l: {:.3f} psudo_acc {:.2f} gpu {}'.format(
                args.s + "2" + args.t, epoch + 1, args.max_epoch_target, acc * 100, acc_val * 100,
                np.mean(total_losses), np.mean(classifier_losses), np.mean(unl_losses), np.mean(ws),
                np.mean(entropy_losses), np.mean(ls),psudo_acc,args.gpu_id)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

        scheduler.step(np.mean(classifier_losses))

        # print("unl_w", args.unl_w)
        if max_pred_acc<acc_val:
            best_F=deepcopy(netF)
            bestC=deepcopy(netC)
            max_pred_acc=acc_val

        if best_test_acc<acc:
            best_test_acc=acc

        if epoch==0:
            first_epoch_acc=acc
        # if acc<first_epoch_acc-0.15:# stop the program if acc drop down.
        #     var = vars(args)
        #     var["val_acc"] = max_pred_acc
        #     var["test_acc"] = acc
        #     var["best_test_acc"] = best_test_acc
        #     # write_shared_file("run_officehome.txt",[args.out_file.name+': val_acc:{:.2f}, test_acc:{:.2f} \n'.format(max_pred_acc,acc)])
        #     write_csv_file("run_office-home_2022.05.17.csv", var)
        #     return

            # max_cluter_acc=cluster_acc
    name='tar_' + args.s+"2"+args.t+"_lr"+str(args.lr)+ "MNPC"+str(args.max_num_per_class)+ "_im"+str(args.im)+ "_u"+str(args.lambda_u)+"_unlent"+str(args.unlent)+"_nonlinear"+str(args.nonlinear)+"_wnl"+str(args.wnl)+"_alpha"+str(args.alpha)+"_num"+str(args.num)+"_"

    torch.save(best_F, osp.join(args.output_dir, name+"target_F.pt"))
    # torch.save netB.state_dict(), osp.join(args.output_dir, "target_B.pt"))
    torch.save(bestC, osp.join(args.output_dir, name+"target_C.pt"))

    acc, _  = cal_acc(target_loader_test, best_F, bestC)
    log_str="test_acc: {:.4f} ".format(acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')


    # file_name1 = osp.join(args.output_dir, 'tar_' + args.s + "2" + args.t + "_lr" + str(args.lr)
    #                         + "MNPC" + str(args.max_num_per_class) + "_im" + str(args.im)
    #                         + "_u" + str(args.lambda_u) + "_unlent" + str(args.unlent) + "_num" + str(
    #     args.num) + 'embeding_adaptation.tsv')
    # file_name2 = osp.join(args.output_dir, 'tar_' + args.s + "2" + args.t + "_lr" + str(args.lr)
    #                         + "MNPC" + str(args.max_num_per_class) + "_im" + str(args.im)
    #                         + "_u" + str(args.lambda_u) + "_unlent" + str(args.unlent) + "_num" + str(
    #     args.num) + 'meta_adaptation.tsv')
    # np.savetxt(file_name1, emb, delimiter='\t')
    # np.savetxt(file_name2, labels, delimiter='\t')


    var=vars(args)
    var["val_acc"]=max_pred_acc
    var["test_acc"]=acc
    var["best_test_acc"] = best_test_acc
    write_csv_file("run_office-home_2022.05.17.csv",var)
    return netF, netC

def Entropy(input_):
    # bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

    def _l2_normalize(d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, netF,netC, x):
        # print(torch.mean(x))
        with torch.no_grad():
            pred = F.softmax(netC(netF(x)), dim=1)
            # softmax_out = nn.Softmax(dim=1)(pred)
            # entropy = Entropy(softmax_out)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        # print("d",torch.mean(d))
        with _disable_tracking_bn_stats(netF):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = netC(netF(x + self.xi * d))
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                netF.zero_grad()
                netC.zero_grad()

            # calc LDS
            # r_adv = d * self.eps
            # print("d, entropy", d.shape, torch.mean(entropy))
            # entropy=entropy.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 3, 224, 224)
            # entropy=torch.sigmoid(entropy)

            # r_adv = d * self.eps*(10/torch.exp(entropy)
            r_adv = d * self.eps
            # r_adv = d * self.eps*entropy
            pred_hat = netC(netF((x + r_adv)))
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

def test_target_svd(args):
    from collections import OrderedDict
    source_loader, _,_, target_loader_unl, _, target_loader_test, class_list = return_dataset(args)
    netF, netC, _ = get_model(args)
    name='tar_' + args.s+"2"+args.t+"_lr"+str(args.lr)+ "MNPC"+str(args.max_num_per_class)+ "_im"+str(args.im)+ "_u"+str(args.lambda_u)+"_unlent"+str(args.unlent)+"_nonlinear"+str(args.nonlinear)+"_wnl"+str(args.wnl)+"_alpha"+str(args.alpha)+"_num"+str(args.num)+"_"

    if args.nonlinear==1:
        args.modelpath = args.output_dir + '/target_F.pt'
    else:
        args.modelpath = args.output_dir + '/{}target_F.pt'.format(name)

    F_dic=torch.load(args.modelpath)
    new_F_dic=OrderedDict()
    for k,v in F_dic.state_dict().items():
        mname=k.replace('module.', '')
        new_F_dic[mname]=v 
       
    netF.load_state_dict(new_F_dic)

    if args.nonlinear==1:
        args.modelpath = args.output_dir + '/target_C.pt'
    else:
        args.modelpath = args.output_dir + '/{}target_C.pt'.format(name)

    C_dic=torch.load(args.modelpath)
    new_C_dic=OrderedDict()
    for k,v in C_dic.state_dict().items():
        mname=k.replace('module.', '')
        new_C_dic[mname]=v 
       
    netC.load_state_dict(new_C_dic)



    netC=netC.cuda()
    netF=netF.cuda()
    netF.eval()
    netC.eval()

    all_fea,all_output,all_label=compute_stride(target_loader_unl,netF,netC,args,istarget=True)
    # print("ddddd")
    len_data=len(all_fea)
    bs=args.batch_size

    all_fea=all_fea.reshape(-1,bs,all_fea.shape[1])
    print('begin',all_fea.shape)
    u,s,v= torch.linalg.svd(all_fea.cuda().permute(0,2,1))
    print("done",u.shape,s.shape,v.shape)

    target_S= torch.mean(s,dim=0)
    target_U= torch.mean(u,dim=0)

    all_fea,all_output,all_label=compute_stride(source_loader,netF,netC,args)
    len_data=len(all_fea)
    bs=args.batch_size

    all_fea=all_fea.reshape(-1,bs,all_fea.shape[1])
    print('begin',all_fea.shape)
    u,s,v= torch.linalg.svd(all_fea.cuda().permute(0,2,1))
    print("done",u.shape,s.shape,v.shape)

    source_S= torch.mean(s,dim=0)
    source_U= torch.mean(u,dim=0)

    p_s, cospa, p_t = torch.svd(torch.mm(source_U.t(), target_U))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    subspace_distance=torch.norm(sinpa,1)
    source_S=source_S.reshape(-1).cpu().numpy()
    target_S=target_S.reshape(-1).cpu().numpy()
    # print("subspace_distance: ", subspace_distance)
    
    log_str = "subspace_distance:{:.4f}".format(subspace_distance)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')


    
    # source_path=args.output_dir+"/"+name+args.s+".csv"
    # with open(file_path,'w',encoding='utf-8') as f:

    np.savetxt(args.output_dir+"/"+name+args.s+".csv",source_S,delimiter=",")
    np.savetxt(args.output_dir+"/"+name+args.t+".csv",target_S,delimiter=",")

def compute_stride(loader,netF,netC,args,istarget=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            if i%5==0:

                if istarget:
                    data, _ = iter_test.next()
                else :
                    data = iter_test.next()

                inputs = data[0]
                # labels = data[1]
                inputs = inputs.cuda()
                feas = netF(inputs)
                outputs = netC(feas)
                # yield (feas,outputs,labels)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    # all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    # all_label = torch.cat((all_label, labels.float()), 0)

    # all_output = nn.Softmax(dim=1)(all_output)
    # # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # # unknown_weight = 1 - ent / np.log(args.class_num)
    # _, predict = torch.max(all_output, 1)
    #
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    #
    # print("pred_accuracylabel: ",accuracy)
    return all_fea,all_output,None

def compute(loader,netF,netC,args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            # yield (feas,outputs,labels)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # all_output = nn.Softmax(dim=1)(all_output)
    # # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # # unknown_weight = 1 - ent / np.log(args.class_num)
    # _, predict = torch.max(all_output, 1)
    #
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    #
    # print("pred_accuracylabel: ",accuracy)
    return all_fea,all_output,all_label


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets,T=1.0):
        log_probs = self.logsoftmax(inputs/T)

        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # print((targets * log_probs).mean(0).shape)
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

def cal_acc(loader, netF, netC):
    start_test = True
    # with torch.cuda.amp.autocast:
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels=labels.cuda()#2020 07 06
            # inputs = inputs
            outputs= netC(netF(inputs))
            # outputs,margin_logits = netC(netF(inputs),labels)
            labels=labels.cpu()#2020 07 06
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent

def cal_acc11111111111111111(loader, netF, netC):
# from tensorboardX import SummaryWriter
# writer = SummaryWriter(log_dir='./logs_kk', comment='cat image')  # 这里的logs要与--logdir的参数一样
    import random
    import numpy as np
    start_test = True
    # with torch.cuda.amp.autocast:
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels=labels.cuda()#2020 07 06
            # inputs = inputs
            embs=netF(inputs)
            outputs= netC(embs)
            # outputs,margin_logits = netC(netF(inputs),labels)
            labels=labels.cpu()#2020 07 06
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_embs=embs.float().cpu()
                start_test = False
            else:
                all_embs= torch.cat((all_embs, embs.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    index=random.sample(list(range(len(all_label))),1000)
    # print(all_embs.shape,all_label.shape)
    # writer.add_embedding(
    #     all_embs[index,:],
    #     metadata=all_label[index],
    #     # label_img=all_label
    # )
    # writer.close()  # 执行close立即刷新，否则将每120秒自动刷新

    return accuracy, mean_ent,all_embs[index,:].cpu().numpy(),all_label[index].cpu().numpy()


def get_model(args):
    netF,netC,netD=None,None,None
    if args.net == 'resnet34':
        netF = resnet34(args=args)
        inc = args.bottleneck
        netC=Predictor(num_class=args.class_num,inc=inc,norm_feature=args.norm_feature,temp=args.temp)
        # netC=LMCL_loss(args.class_num, inc, s=1.00, m=0.4)
    elif args.net == 'resnet50':
        netF = resnet50(args=args)
        inc = args.bottleneck
        netC=Predictor(num_class=args.class_num,inc=inc,norm_feature=args.norm_feature,temp=args.temp)
        # netC=LMCL_loss(args.class_num, inc, s=1.00, m=0.4)
    elif args.net == "alexnet":
        inc = args.bottleneck
        netF = AlexNetBase(bootleneck_dim=inc)
        netC = Predictor(num_class=args.class_num, inc=inc,norm_feature=args.norm_feature,temp=args.temp)
        # netC = Predictor(num_class=args.class_num, inc=inc)
    elif args.net == "vgg":
        inc = args.bottleneck
        netF = VGGBase(bootleneck_dim=inc)
        # inc = 25088
        netC = Predictor(num_class=args.class_num, inc=inc,norm_feature=args.norm_feature,temp=args.temp)
    else:
        raise ValueError('Model cannot be recognized.')
    print(get_para_num(netF))
    print(get_para_num(netC))

    return  netF,netC,netD

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=int, default=1, help="device id to run")
    parser.add_argument('--net', type=str, default="vgg", choices=['vgg',"alexnet","resnet34",'LeNet', 'resnet50', 'ResNet101'])
    parser.add_argument('--s', type=str, default="Real_World", help="source office_home :Art Clipart Product Real_World")
    parser.add_argument('--t', type=str, default="Product", help="target  Art Clipart Product Real_World")
    parser.add_argument('--max_epoch_source', type=int, default=30, help="maximum epoch ")
    # parser.add_argument('--skd_src', type=int, default=1, help="maximum epoch ")
    parser.add_argument('--max_epoch_target', type=int, default=30, help="maximum epoch ")
    parser.add_argument('--num', type=int, default=1, help="labeled_data per class")
    parser.add_argument('--train', type=int, default=1, help="if to train")
    # parser.add_argument('--adv', type=int, default=0, help="if to adversarial")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--class_num', type=int, default=65, help="batch_size",choices=[65,10,31,126])
    parser.add_argument('--worker', type=int, default=16, help="number of workers")
    # parser.add_argument('--dset', type=str, default='u2m', choices=['u2m', 'm2u', 's2m'])
    parser.add_argument('--dataset', type=str, default='office-home', choices=['office-home', 'multi','digits', 'Office-31'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--update_cls', type=int, default=0, help="random seed")
    parser.add_argument('--max_num_per_class', type=int, default=0, help="random seed")
    parser.add_argument('--norm_feature', type=int, default=0, help="random seed")
    parser.add_argument('--par', type=float, default=1)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--lambda_u', type=float, default=200)
    parser.add_argument('--im', type=float, default=1)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='2021_02_03Office-31')
    parser.add_argument('--epsilon', type=float, default=1e-5)
    # parser.add_argument('--lent', type=float, default=0.0)
    parser.add_argument('--unlent', type=float, default=1)
    parser.add_argument('--unl_w', type=float, default=0)
    parser.add_argument('--vat_w', type=float, default=0)
    parser.add_argument('--div_w', type=float, default=1)
    parser.add_argument('--uda', type=int, default='0', choices=[0, 1])
    parser.add_argument('--nonlinear', type=int, default='0', choices=[0, 1])
    parser.add_argument('--wnl', type=int, default='0', choices=[0, 1])
    args = parser.parse_args()

    # print("uda",args.uda)
    # args.max_epoch_target=100
    # set_gpu(args.gpu_id)
    # if args.lambda_u>200:
    #     exit()
    args.gpu_id=getAvaliableDevice(min_mem=24000)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # print("use GPU",args.gpu_id)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
    setup_seed(args.seed)
    #torch.backends.cudnn.benchmark = True

    import warnings
    warnings.filterwarnings("ignore")
    current_folder = "./ssda"
    # current_folder = "./"
    args.output_dir = osp.join(current_folder, args.output, 'seed' + str(args.seed), args.dataset,args.s+"_norm"+str(args.norm_feature)+"_temp"+str(args.temp)+"_lr"+str(args.lr))
    # args.output_dir = osp.join(current_folder, args.output, 'seed' + str(args.seed), args.dataset,args.s+"_norm"+str(args.norm_feature)+"_temp"+str(args.temp))
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # with torch.cuda.device(args.gpu_id):
    # args.lr = 0.0001

    # test_target_svd(args)
    
    if args.train==1:
        args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)

    else:
        args.out_file = open(osp.join(args.output_dir, 'tar_' + args.s+"2"+args.t+"_lr"+str(args.lr)
                                        + "MNPC"+str(args.max_num_per_class)+ "_im"+str(args.im)
                                        + "_u"+str(args.lambda_u)+"_unlent"+str(args.unlent)
                                        +"_nonlinear"+str(args.nonlinear)+"_wnl"+str(args.wnl)+"_alpha"+str(args.alpha)
                                        +"_num"+str(args.num)+'.txt'), 'w')
        test_target(args)
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
        test_target_svd(args)
