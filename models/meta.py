import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch import optim
import  numpy as np
from scipy.spatial.distance import cdist
from loss import CrossEntropyLabelSmooth
from tqdm import tqdm
from loss import Entropy
from copy import deepcopy
class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self,encoder,classifier, feat_bootleneck,config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = config["inner_lr"]
        self.meta_lr = config["outer_lr"]
        self.n_way = config["class_num"]
        # self.k_spt = config["train_shot"]
        # self.k_qry = config["train_query"]
        # self.task_num = args.task_num
        self.update_step = config["update_step"]
        self.clip=config["grad_clip"]
        self.update_step_test = config["step_test"]
        # self.device=config["device"]
        self.encoder = encoder
        self.classifier=classifier
        self.global_center=None
        self.feat_bootleneck=feat_bootleneck
        self.config=config
        self.ft_lr=config["ft_lr"]
        param_group = []
        learning_rate = self.meta_lr
        for k, v in self.encoder.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.feat_bootleneck.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.classifier.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]
        self.meta_optim = optim.SGD(param_group, momentum=0.9,  nesterov=True, weight_decay=config["weight_decay"])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.meta_optim, mode='min',
                                                                    factor=0.2, patience=config["patience"],
                                                                  verbose=True, min_lr=1e-6)
        self.task_index=1
        self.update_flag=config["batch_per_episodes"]
        # self.loss=nn.CrossEntropyLoss()
        self.loss=CrossEntropyLabelSmooth(self.n_way,device=None)
        self.cluster=False

    def obtain_center(self,data, netF, netC, netB=None):
        start_test = True
        with torch.no_grad():
            inputs = data[0]
            labels = data[1]
            feas = netB(netF(inputs))
            outputs = netC(feas)
            all_fea = feas.float().cpu()
            all_output = outputs.float().cpu()
            all_label = labels.float().cpu()

            all_output = nn.Softmax(dim=1)(all_output)
            _, predict = torch.max(all_output, 1)
            accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
            all_fea = all_fea.float().cpu().numpy()

            K = all_output.size(1)
            aff = all_output.float().cpu().numpy()
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            center = torch.from_numpy(initc).cuda()

            log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            # print(log_str + '\n')
        return center

    def obtain_label(self,features_target, center,device=None):
        features_target = torch.cat((features_target, torch.ones(features_target.size(0), 1).cuda()), 1)
        fea = features_target.float().detach().cpu().numpy()
        center = center.float().detach().cpu().numpy()
        dis = cdist(fea, center, 'cosine') + 1
        pred = np.argmin(dis, axis=1)
        pred = torch.from_numpy(pred).cuda()
        return pred

    def forward(self, support_data, query_data):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """

        [support_image, support_label] = support_data
        [query_image, query_label] = query_data

        task_num = 1

        querysz = query_label.size()[0]

        losses_q = [0 for _ in range(self.update_step)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step)]

        for i in range(task_num):

            fast_parameters = list(self.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.parameters():
                weight.fast = None
            self.zero_grad()

            for k in range(0, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                # net_F,net_C=deepcopy(self.encoder),deepcopy(self.classifier)
                # center = self.obtain_center(support_data, net_F,net_C)

                if self.cluster is True:
                    center = self.obtain_center(support_data, self.encoder,netC=self.classifier,netB=self.feat_bootleneck)
                    with torch.no_grad():
                        # features_support = net_F(support_image)
                        features_support = self.feat_bootleneck(self.encoder(support_image))
                        pred = self.obtain_label(features_support, center,None)

                    logits= self.classifier(self.feat_bootleneck(self.encoder(support_image)))
                    loss = self.loss(logits,pred)


                else:
                    logits = self.classifier(self.feat_bootleneck(self.encoder(support_image)))

                    loss = self.loss(logits, support_label)
                # buiuld graph supld fport gradient of gradient
                grad = torch.autograd.grad(loss, fast_parameters,create_graph=True)

                fast_parameters = []

                for index, weight in enumerate(self.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    # if grad[k] is None:

                    #     fast_parameters.append(weight.fast)
                    #     continue
                    if weight.fast is None:
                        weight.fast = weight - self.update_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.update_lr * grad[index]

                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)

                logits_q = self.classifier(self.feat_bootleneck(self.encoder(query_image)))
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.loss(logits_q, query_label)

                losses_q[k] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label).sum().item()  # convert to numpy
                    corrects[k] = corrects[k] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # print("loss",loss_q.item())
        # optimize theta parameters
        # self.meta_optim.zero_grad()
        loss_q.backward()
        # self.meta_optim.step()

        if self.task_index==self.update_flag:
            if self.clip > 0.1:  # 0.1 threshold wether to do clip
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
            self.meta_optim.step()
            self.meta_optim.zero_grad()
            self.task_index=1
        else:
            self.task_index=self.task_index+1

        accs = 100*np.array(corrects) / (querysz * task_num)

        return accs,loss_q.item()

    def finetunning(self, support_data, query_data):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        [support_image, support_label] = support_data
        [query_image, query_label] = query_data

        task_num = 1

        # querysz = query_label.size()[0]

        # losses_q = [0 for _ in range(self.update_step)]  # losses_q[i] is the loss on step i
        # corrects = [0 for _ in range(self.update_step)]

        querysz = query_label.size()[0]

        losses_q = [0 for _ in range(self.update_step_test)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step_test)]
        for i in range(task_num):
            fast_parameters = list(
                self.parameters())  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.parameters():
                weight.fast = None
            self.zero_grad()


            for k in range(0, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1

                netF=deepcopy(self.encoder)
                netB=deepcopy(self.feat_bootleneck)
                # net
                if self.global_center  is not None:
                    center=self.global_center
                else:
                    data=[support_image,support_label]
                    center = self.obtain_center(data, netF=self.encoder, netC=self.classifier,netB=self.feat_bootleneck)

                with torch.no_grad():
                    # features_support = net_F(support_image)
                    features_support = self.encoder(support_image)
                    pred = self.obtain_label(features_support, center,device=None)
                logits = self.classifier(self.encoder(support_image))

                loss = self.loss(logits, pred)

                # buiuld graph supld fport gradient of gradient
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)

                fast_parameters = []

                for index, weight in enumerate(self.parameters()):
                    # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                    # if grad[k] is None:
                    #     fast_parameters.append(weight.fast)
                    #     continue
                    if weight.fast is None:
                        weight.fast = weight - self.update_lr * grad[index]  # create weight.fast
                    else:
                        # create an updated weight.fast,
                        # note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.update_lr * grad[index]

                    # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                    fast_parameters.append(weight.fast)
                    # print('add')
                logits_q= self.classifier(self.encoder(query_image))
                # # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.loss(logits_q, query_label)

                losses_q[k] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, query_label).sum().item()  # convert to numpy
                    corrects[k] = corrects[k] + correct

        accs = 100*np.array(corrects) / querysz*task_num

        return accs,losses_q

    def finetunning_shot(self, target_dataset):
        # fast_parameters = list(self.parameters())  # the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        ## set base network
        netF=deepcopy(self.encoder)
        netC =deepcopy( self.classifier)
        netB=deepcopy(self.feat_bootleneck)

        for k, v in netC.named_parameters():
            v.requires_grad = False
        param_group = []
        for k, v in netF.named_parameters():
            param_group += [{'params': v, 'lr': self.ft_lr}]
        for k, v in netB.named_parameters():
            param_group += [{'params': v, 'lr': self.ft_lr}]
        optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in tqdm(range(self.config["ft_epoch"]), leave=False):
            netF.train()
            netB.train()
            # iter_test = iter(dset_loaders["target"])
            im_loss, classifier_loss = None, None
            prev_F = deepcopy(netF)
            prev_B = deepcopy(netB)
            prev_F.eval()
            prev_B.eval()
            center = self.build_global_center(target_dataset,prev_F,prev_B,netC)
            for _, data in tqdm(enumerate(target_dataset), leave=False):
                # if inputs_test.size(0) == 1:
                #     continue
                inputs_test = data['T'].cuda()
                with torch.no_grad():
                    features_test = prev_B(prev_F(inputs_test))
                    pred = self.obtain_label(features_test, center)

                features_test = netB(netF(inputs_test))
                outputs_test = netC(features_test)
                classifier_loss = CrossEntropyLabelSmooth(num_classes=self.config["class_num"], epsilon=0)(outputs_test, pred)

                softmax_out = nn.Softmax(dim=1)(outputs_test)
                im_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                im_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                total_loss = im_loss + 0.1 * classifier_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            netF.eval()
            netB.eval()
            # netC.eval()
            acc, _ = self.cal_acc(target_dataset, netF, netB, netC)
            log_str = 'tra-tag to: {}, Iter:{}/{}; Accuracy = {:.2f}%;t_ls: {:.4f}; mi_ls:{:.4f}: cl_loss: {:.4f}'\
                .format(self.config["target"], epoch + 1,
                self.config["ft_epoch"], acc * 100,total_loss.item(),im_loss.item(),classifier_loss.item())
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            print(log_str + '\n')

        for k, v in netC.named_parameters():
            v.requires_grad = True
        # torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F.pt"))
        # torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B.pt"))
        # torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C.pt"))
    def build_global_center(self,loader,prev_F,prev_B,netC):
        netF=prev_F
        netC=netC
        netB=prev_B
        start_test = True
        with torch.no_grad():
            for i, data in enumerate(loader, 1):
                # data = iter_test.next()
                inputs = data['T']
                labels = data['T_label']
                inputs = inputs.cuda()
                feas = netB(netF(inputs))
                outputs = netC(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        center = torch.from_numpy(initc).cuda()

        log_str = 'predict acc to cluster acc = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        # args.out_file.write(log_str + '\n')
        # args.out_file.flush()
        print(log_str + '\n')
        # self.global_center=center
        return center

    def remove_global_center(self):
        self.global_center=None
    def adapt_meta_learning_rate(self,loss):
        self.scheduler.step(loss)
    def get_meta_learning_rate(self):
        epoch_learning_rate=[]
        for param_group in self.meta_optim.param_groups:
            epoch_learning_rate.append(param_group['lr'])
        return epoch_learning_rate[0]

    def cal_acc(self,loader, netF, netB, netC):
        start_test = True
        with torch.no_grad():
            # iter_test = iter(loader)
            for i, data in enumerate(loader, 1):
                # data = iter_test.next()
                inputs = data['T']
                labels = data['T_label']
                inputs = inputs.cuda()
                outputs = netC(netB(netF(inputs)))
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
if __name__ == '__main__':
    pass
