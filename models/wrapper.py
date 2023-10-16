"""Meta-learners for Omniglot experiment.
Based on original implementation:
https://github.com/amzn/metalearn-leap
"""
import random
from abc import abstractmethod
from torch import nn
from torch import optim
import torch
import numpy as np

from copy import deepcopy
from scipy.spatial.distance import cdist
import warpgrad
from leap import Leap
from leap.utils import clone_state_dict

from models.warpmain_utils import Res, AggRes


class BaseWrapper(object):

    """Generic training wrapper.

    Arguments:
        criterion (func): loss criterion to use.
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
    """

    def __init__(self, criterion, model, optimizer_cls, optimizer_kwargs):
        self.criterion = criterion
        self.model = model

        self.optimizer_cls = \
            optim.SGD if optimizer_cls.lower() == 'sgd' else optim.Adam
        self.optimizer_kwargs = optimizer_kwargs

    def __call__(self, tasks, meta_train=True):
        return self.run_tasks(tasks, meta_train=meta_train)

    @abstractmethod
    def _partial_meta_update(self, loss, final):
        """Meta-model specific meta update rule.

        Arguments:
            loss (nn.Tensor): loss value for given mini-batch.
            final (bool): whether iteration is the final training step.
        """
        NotImplementedError('Implement in meta-learner class wrapper.')

    @abstractmethod
    def _final_meta_update(self):
        """Meta-model specific meta update rule."""
        NotImplementedError('Implement in meta-learner class wrapper.')

    def run_tasks(self, tasks, meta_train):
        """Train on a mini-batch tasks and evaluate test performance.

        Arguments:
            tasks (list, torch.utils.data.DataLoader): list of task-specific
                dataloaders.
            meta_train (bool): whether current run in during meta-training.
        """
        # assert len(tasks)==2
        results = []
        # print(tasks)
        for task in tasks:
            # print("task",task)
            task.dataset.train()
            trainres = self.run_task(task, train=True, meta_train=meta_train)
            # task.dataset.eval()
            # valres = self.run_task(task, train=False, meta_train=False)
            results.append(trainres)
            # break # the first task is unsupervised training
        ##
        results = AggRes(results)

        # Meta gradient step
        if meta_train:
            self._final_meta_update()

        return results

    def run_task(self, task, train, meta_train):
        """Run model on a given task.

        Arguments:
            task (torch.utils.data.DataLoader): task-specific dataloaders.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        optimizer = None
        if train:
            self.model.init_adaptation()
            self.model.train()
            optimizer = self.optimizer_cls(
                self.model.parameters(), **self.optimizer_kwargs)
        else:
            self.model.eval()

        return self.run_batches(
            task, optimizer, train=train, meta_train=meta_train)

    def build_global_center(self,batches,device):
        start_test = True
        with torch.no_grad():
            for n, (inputs, labels) in enumerate(batches):
                # data = iter_test.next()
                inputs = inputs.to(device)
                feas,logits = self.model(inputs)
                outputs = logits
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
        return center,pred_label

    def obtain_center(self,data):
        start_test = True
        with torch.no_grad():
            inputs = data[0]
            labels = data[1]
            feas,logits = self.model(inputs)
            outputs = logits
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

            log_str = 'predict_acc = {:.2f}% -> cluster_acc {:.2f}%'.format(accuracy * 100, acc * 100)
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


    def run_batches(self, batches, optimizer, train=False, meta_train=False):
        """Iterate over task-specific batches.

        Arguments:
            batches (torch.utils.data.DataLoader): task-specific dataloaders.
            optimizer (torch.nn.optim): optimizer instance if training is True.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        device = next(self.model.parameters()).device
        # t1,t2=[],[]
        # for n, (input, target) in enumerate(batches):
        #     t1.append(target)
        # for n, (input, target) in enumerate(batches):
        #     t2.append(target)
        # print("t1",t1,"\nt2",t2)

        res = Res()
        N = len(batches)
        center, _= self.build_global_center(batches,device)


        optimizer.zero_grad()
        for n, (input, target) in enumerate(batches):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Evaluate model
            _,prediction = self.model(input)
            
            with torch.no_grad():
                features_support,_ = self.model(input)
                pred = self.obtain_label(features_support, center,None)

            loss = self.criterion(prediction, pred)

            res.log(loss=loss.item(), pred=prediction, target=target)

            # TRAINING #
            if not train:
                continue

            # final = (n+1) == N
            loss.backward()

            # if meta_train:
            #     self._partial_meta_update(loss, final)

            optimizer.step()
            optimizer.zero_grad()

            # if final:
            #     break
        ###
        res.aggregate()
        return res

def get_para_num(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
class WarpGradWrapper(BaseWrapper):

    """Wrapper around WarpGrad meta-learners.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon
            construction.
        meta_kwargs (dict): kwargs to pass to meta-learner upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self,
                 model,
                 optimizer_cls,
                 meta_optimizer_cls,
                 optimizer_kwargs,
                 meta_optimizer_kwargs,
                 meta_kwargs,
                 criterion):

        replay_buffer = warpgrad.ReplayBuffer(
            inmem=meta_kwargs.pop('inmem', True),
            tmpdir=meta_kwargs.pop('tmpdir', None))

        optimizer_parameters = warpgrad.OptimizerParameters(
            trainable=meta_kwargs.pop('learn_opt', False),
            default_lr=optimizer_kwargs['lr'],
            default_momentum=optimizer_kwargs['momentum']
            if 'momentum' in optimizer_kwargs else 0.)

        updater = warpgrad.DualUpdater(criterion, **meta_kwargs)
        # p = get_para_num(model)
        # print("before warp",p)
        model = warpgrad.Warp(model=model,
                              adapt_modules=list(model.adapt_modules()),
                              warp_modules=list(model.warp_modules()),
                              updater=updater,
                              buffer=replay_buffer,
                              optimizer_parameters=optimizer_parameters)
        # p = get_para_num(model)
        # print("after warp", p)
        # total_num = sum(p.numel() for p in model.parameters())
        # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print('Total', total_num, 'Trainable', trainable_num)
        super(WarpGradWrapper, self).__init__(criterion,
                                              model,
                                              optimizer_cls,
                                              optimizer_kwargs)

        self.meta_optimizer_cls = optim.SGD \
            if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        lra = meta_optimizer_kwargs.pop(
            'lr_adapt', meta_optimizer_kwargs['lr'])
        lri = meta_optimizer_kwargs.pop(
            'lr_init', meta_optimizer_kwargs['lr'])
        lrl = meta_optimizer_kwargs.pop(
            'lr_lr', meta_optimizer_kwargs['lr'])
        self.meta_optimizer = self.meta_optimizer_cls(
            [{'params': self.model.init_parameters(), 'lr': lri},
             {'params': self.model.warp_parameters(), 'lr': lra},
             {'params': self.model.optimizer_parameters(), 'lr': lrl}],
            **meta_optimizer_kwargs)

    def _partial_meta_update(self, loss, final):
        pass

    def _final_meta_update(self):

        def step_fn():
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        self.model.backward(step_fn, **self.optimizer_kwargs)

    def run_task(self, task, train, meta_train):
        """Run model on a given task, first adapting and then evaluating"""
        if meta_train and train:
            # Register new task in buffer.
            self.model.register_task(task)
            self.model.collect()
        else:
            # Make sure we're not collecting non-meta-train data
            self.model.no_collect()

        optimizer = None
        if train:

            # Initialize model adaptation
            self.model.init_adaptation()

            optimizer = self.optimizer_cls(
                self.model.optimizer_parameter_groups(),
                **self.optimizer_kwargs)

            if self.model.collecting and self.model.learn_optimizer:
                # Register optimiser to collect potential momentum buffers
                self.model.register_optimizer(optimizer)
        else:
            self.model.eval()

        return self.run_batches(
            task, optimizer, train=train, meta_train=meta_train)


class LeapWrapper(BaseWrapper):

    """Wrapper around the Leap meta-learner.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon
            construction.
        meta_kwargs (dict): kwargs to pass to meta-learner upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self,
                 model,
                 optimizer_cls,
                 meta_optimizer_cls,
                 optimizer_kwargs,
                 meta_optimizer_kwargs,
                 meta_kwargs,
                 criterion):
        super(LeapWrapper, self).__init__(criterion,
                                          model,
                                          optimizer_cls,
                                          optimizer_kwargs)
        self.meta = Leap(model, **meta_kwargs)

        self.meta_optimizer_cls = \
            optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        self.meta_optimizer = self.meta_optimizer_cls(
            self.meta.parameters(), **meta_optimizer_kwargs)

    def _partial_meta_update(self, l, final):
        self.meta.update(l, self.model)

    def _final_meta_update(self):
        self.meta.normalize()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()

    def run_task(self, task, train, meta_train):
        if meta_train:
            self.meta.init_task()

        if train:
            self.meta.to(self.model)

        return super(LeapWrapper, self).run_task(task, train, meta_train)


class MAMLWrapper(object):

    """Wrapper around the MAML meta-learner.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon
            construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self, model, optimizer_cls, meta_optimizer_cls,
                 optimizer_kwargs, meta_optimizer_kwargs, criterion):
        self.criterion = criterion
        self.model = model

        self.optimizer_cls = \
            maml.SGD if optimizer_cls.lower() == 'sgd' else maml.Adam

        self.meta = maml.MAML(optimizer_cls=self.optimizer_cls,
                              criterion=criterion,
                              model=model,
                              tensor=False,
                              **optimizer_kwargs)

        self.meta_optimizer_cls = \
            optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam

        self.optimizer_kwargs = optimizer_kwargs
        self.meta_optimizer = self.meta_optimizer_cls(self.meta.parameters(),
                                                      **meta_optimizer_kwargs)

    def __call__(self, meta_batch, meta_train):
        tasks = []
        for t in meta_batch:
            t.dataset.train()
            inner = [b for b in t]
            t.dataset.train()
            outer = [b for b in t]
            tasks.append((inner, outer))
        return self.run_meta_batch(tasks, meta_train=meta_train)

    def run_meta_batch(self, meta_batch, meta_train):
        """Run on meta-batch.

        Arguments:
            meta_batch (list): list of task-specific dataloaders
            meta_train (bool): meta-train on batch.
        """
        loss, results = self.meta(meta_batch,
                                  return_predictions=False,
                                  return_results=True,
                                  create_graph=meta_train)
        if meta_train:
            loss.backward()
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        return results


class NoWrapper(BaseWrapper):

    """Wrapper for baseline without any meta-learning.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        criterion (func): loss criterion to use.
    """
    def __init__(self, model, optimizer_cls, optimizer_kwargs, criterion):
        super(NoWrapper, self).__init__(criterion,
                                        model,
                                        optimizer_cls,
                                        optimizer_kwargs)
        self._original = clone_state_dict(model.state_dict(keep_vars=True))

    def __call__(self, tasks, meta_train=False):
        return super(NoWrapper, self).__call__(tasks, meta_train=False)

    def run_task(self, task, train, meta_train):
        if train:
            self.model.load_state_dict(self._original)
        return super(NoWrapper, self).run_task(task, train, meta_train)

    def _partial_meta_update(self, loss, final):
        pass

    def _final_meta_update(self):
        pass


class _FOWrapper(BaseWrapper):

    """Base wrapper for First-order MAML and Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon
            construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = None

    def __init__(self, model, optimizer_cls, meta_optimizer_cls,
                 optimizer_kwargs, meta_optimizer_kwargs, criterion):
        super(_FOWrapper, self).__init__(criterion,
                                         model,
                                         optimizer_cls,
                                         optimizer_kwargs)
        self.meta_optimizer_cls = \
            optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        self.meta_optimizer_kwargs = meta_optimizer_kwargs

        self._counter = 0
        self._updates = None
        self._original = clone_state_dict(
            self.model.state_dict(keep_vars=True))

        params = [p for p in self._original.values()
                  if getattr(p, 'requires_grad', False)]
        self.meta_optimizer = self.meta_optimizer_cls(params,
                                                      **meta_optimizer_kwargs)

    def run_task(self, task, train, meta_train):
        if meta_train:
            self._counter += 1
        if train:
            self.model.load_state_dict(self._original)
        return super(_FOWrapper, self).run_task(task, train, meta_train)

    def _partial_meta_update(self, loss, final):
        if not final:
            return

        if self._updates is None:
            self._updates = {}
            for n, p in self._original.items():
                if not getattr(p, 'requires_grad', False):
                    continue

                if p.size():
                    self._updates[n] = p.new(*p.size()).zero_()
                else:
                    self._updates[n] = p.clone().zero_()

        for n, p in self.model.state_dict(keep_vars=True).items():
            if n not in self._updates:
                continue

            if self._all_grads is True:
                self._updates[n].add_(p.data)
            else:
                self._updates[n].add_(p.grad.data)

    def _final_meta_update(self):
        for n, p in self._updates.items():
            p.data.div_(self._counter)

        for n, p in self._original.items():
            if n not in self._updates:
                continue

            if self._all_grads:
                p.grad = p.data - self._updates[n].data
            else:
                p.grad = self._updates[n]

        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        self._counter = 0
        self._updates = None


class ReptileWrapper(_FOWrapper):

    """Wrapper for Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon
            construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = True

    def __init__(self, *args, **kwargs):
        super(ReptileWrapper, self).__init__(*args, **kwargs)


class FOMAMLWrapper(_FOWrapper):
    """Wrapper for FOMAML.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon
            construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = False

    def __init__(self, *args, **kwargs):
        super(FOMAMLWrapper, self).__init__(*args, **kwargs)


class FtWrapper(BaseWrapper):

    """Wrapper for Multi-headed finetuning.

    This wrapper differs from others in that it blends batches from all tasks
    into a single epoch.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self, model, optimizer_cls, optimizer_kwargs, criterion):
        super(FtWrapper, self).__init__(criterion,
                                        model,
                                        optimizer_cls,
                                        optimizer_kwargs)
        # We use the same inner optimizer throughout
        self.optimizer = self.optimizer_cls(self.model.parameters(),
                                            **self.optimizer_kwargs)

    @staticmethod
    def gen_multitask_batches(tasks, train):
        """Generates one batch iterator across all tasks."""
        iterator_id = 0
        all_batches = []
        for task_id, iterator in tasks:
            if train:
                iterator.dataset.train()
            else:
                iterator.dataset.eval()

            for batch in iterator:
                all_batches.append((iterator_id, task_id, batch))
            iterator_id += 1

        if train:
            random.shuffle(all_batches)

        return all_batches

    def run_tasks(self, tasks, meta_train):
        original = None
        if not meta_train:
            original = clone_state_dict(self.model.state_dict(keep_vars=True))

            # Non-transductive task evaluation for fair comparison
            for module in self.model.modules():
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()

        # Training #
        all_batches = self.gen_multitask_batches(tasks, train=True)
        trainres = self.run_multitask(all_batches, train=True)

        # Eval #
        all_batches = self.gen_multitask_batches(tasks, train=False)
        valres = self.run_multitask(all_batches, train=False)

        results = AggRes(zip(trainres, valres))

        if not meta_train:
            self.model.load_state_dict(original)

        return results

    def _partial_meta_update(self, l, final):
        return

    def _final_meta_update(self):
        return

    def run_multitask(self, batches, train):
        """Train on task in multi-task mode

        This is equivalent to the run_task method but differs in that
        batches are assumed to be mixed from different tasks.
        """
        N = len(batches)

        if train:
            self.model.train()
        else:
            self.model.eval()

        device = next(self.model.parameters()).device

        res = {}
        for n, (iterator_id, task_id, (input, target)) in enumerate(batches):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            prediction = self.model(input, task_id)

            loss = self.criterion(prediction, target)

            if iterator_id not in res:
                res[iterator_id] = Res()

            res[iterator_id].log(loss=loss.item(),
                                 pred=prediction,
                                 target=target)

            # TRAINING #
            if not train:
                continue

            final = (n + 1) == N
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if final:
                break
        ###
        res = [r[1] for r in sorted(res.items(), key=lambda r: r[0])]
        for r in res:
            r.aggregate()

        return res
