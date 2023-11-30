import numpy as np
import torch
import collections
import heapq
from torch.distributions import Uniform, Exponential, Gamma, Laplace

import time
from tqdm import tqdm, trange
from .prompt_utils import flatten_prompt

class RankSGD_Net:
    def __init__(self, prompt, smoothing_para, stepsize, distribution='Gaussian', moment = 0.5, linesearch=False, lr_decay=0.95):
        self.smoothing_para = smoothing_para
        self.stepsize = stepsize
        self.linesearch = linesearch
        self.lr_decay = lr_decay
        self.moment = moment
        self.distribution=distribution

        self.best_para = prompt

        self.mode = "grad_est"

        self.search_direction = [torch.zeros_like(i) for i in self.best_para]
        self.test_para = None
        self.grad_acumulate_step = 0

    def generate_query_parameter(self, num_query):
        num_para = len(self.best_para)
        self.test_para = []
        if self.mode == "grad_est":
            for i in range(num_para):
                size = [num_query, *self.best_para[i].shape]
                if 'Gaussian' in self.distribution:
                    test_d =  torch.randn(num_query, *self.best_para[i].shape) * self.smoothing_para
                elif 'Uniform' in self.distribution:
                    uniform_dist = Uniform(-1, 1)
                    test_d = uniform_dist.sample(size) * self.smoothing_para
                elif 'Exp' in self.distribution:
                    exp_dist = Exponential(1.0)
                    uniform_dist = Uniform(0, 1)
                    uniform_samples = uniform_dist.sample(size)
                    positive_mask = uniform_samples > 0.5
                    negative_mask = ~positive_mask
                    positive_data = exp_dist.sample(size) * positive_mask
                    negative_data = -exp_dist.sample(size) * negative_mask
                    test_d = (positive_data + negative_data) * self.smoothing_para
                elif 'Gamma' in self.distribution:
                    gamma_dist = Gamma(2.0, 1.0)
                    uniform_dist = Uniform(0, 1)
                    uniform_samples = uniform_dist.sample(size)
                    positive_mask = uniform_samples > 0.5
                    negative_mask = ~positive_mask
                    positive_data = gamma_dist.sample(size) * positive_mask
                    negative_data = -gamma_dist.sample(size) * negative_mask
                    test_d = (positive_data + negative_data) * self.smoothing_para
                elif 'Laplace' in self.distribution:
                    laplace_dist = Laplace(0.0, 1.0)
                    test_d = laplace_dist.sample(size) * self.smoothing_para
                else:
                    raise NotImplementedError
                # test_d = torch.randn(num_query, *self.best_para[i].shape) * self.smoothing_para 
                self.test_para.append(torch.unsqueeze(self.best_para[i], dim=0) + test_d.to(device=self.best_para[i].device))

        else:
            assert num_query > 2
            for i in range(num_para):
                test_d = torch.unsqueeze(self.search_direction[i], 0)
                test_z = torch.zeros_like(test_d)
                dims = [1] * (len(test_d.shape)-1)
                test_d = test_d.repeat(num_query-2, *dims)
                scales = torch.from_numpy(np.array([scale for scale in (0.5 ** np.arange(num_query-2))]).reshape(num_query - 2, *dims))
                test_d *= scales.to(device=test_d.device)
                test_d *= self.stepsize
                test_d = torch.cat([test_d, test_z], dim=0)
                self.test_para.append(torch.unsqueeze(self.best_para[i], dim=0) + test_d.to(device=self.best_para[i].device))

        return self.test_para
    def update(self, rank_info):
        if self.mode == "grad_est":
            # compute the gradient with the rank information
            rank_info = [int(r) for r in rank_info]
            test_para_rank = {}
            for t in range(len(self.test_para[0])):
                if (t+1) in rank_info:
                    test_para_rank[t] = rank_info.index(t+1)
                else:
                    test_para_rank[t] = -1

            #rank-based update
            update_direction = [torch.zeros_like(i) for i in self.best_para]

            for tc, tr in test_para_rank.items():
                if tr >= 0:
                    # -2*tr may be problem
                    for i in range(len(self.test_para)):
                        update_direction[i] += (len(self.test_para[i]) - 2*tr) * (self.test_para[i][tc] - self.best_para[i])
                else:
                    for i in range(len(self.test_para)):
                        update_direction[i] += (-len(rank_info)) * (self.test_para[i][tc] - self.best_para[i])

            k = len(rank_info)
            m = len(self.test_para[0])
            for i in range(len(update_direction)):
                update_direction[i] /= k * (k-1) / 2 + k * (m-k)

            # self.search_direction = self.search_direction * self.grad_acumulate_step + update_direction
            # self.grad_acumulate_step += 1
            # self.search_direction /= self.grad_acumulate_step
            for i in range(len(self.best_para)):
                self.search_direction[i] = self.search_direction[i] * self.moment + update_direction[i]

            if self.linesearch == True:
                self.mode = "line_search"
            else:
                for i in range(len(self.best_para)):
                    self.best_para[i] = self.best_para[i] + self.search_direction[i] * self.stepsize

        else:
            best_ind = int(rank_info[0]) - 1
            # if best_ind != (len(self.test_para) - 1):
            for i in range(len(self.best_para)):
                self.best_para[i] = self.test_para[i][best_ind]
                self.grad_acumulate_step = 0
                self.search_direction[i] = torch.zeros_like(self.best_para[i])

            self.mode = "grad_est"

    def best_ret(self):
        # state_dict = collections.OrderedDict()
        # ptr = 0
        # for key in self.model.state_dict():
        #     numel = self.model.state_dict()[key].numel()
        #     state_dict[key] = self.best_para[ptr:ptr + numel].view(self.model.state_dict()[key].size())
        #     ptr += numel

        return self.best_para

    def update_stepsize(self, stepsize=None):
        if stepsize is None:
            self.stepsize = self.stepsize * self.lr_decay
        else:
            self.stepsize = stepsize

class NetworkOracle(object):
    def __init__(self, loss_fcn, model, m, k, weight_decay, offline):
        self.m = m
        self.k = k
        self.weight_decay = weight_decay
        self.loss_fcn = loss_fcn
        self.model = model
        self.offline = offline

    def __call__(self, test_para, start=0):
        losses = []

        for i in range(len(test_para[0])):
            prompt = [j[i] for j in test_para]
            if self.offline:
                return_mean = self.loss_fcn(self.model, prompt, start=start)
            else:
                return_mean = self.loss_fcn(self.model, prompt) * (-1) # larger return mean better

            losses.append(return_mean)

        min_number = heapq.nsmallest(self.k, losses)
        min_ind = []
        smallest = 0
        for i, t in enumerate(min_number):
            index = losses.index(t)
            min_ind.append(index + 1)
            losses[index] = float('inf')

            if i == 0:
                smallest = t


        return min_ind, smallest
    
class TamingTrainer:
    def __init__(self, 
          model, 
          optimizer, 
          m, 
          smooth,
          batch_size, 
          loss_fn,
          distribution='Gaussian', 
          scheduler=None, 
          eval_fns=None, 
          linesearch=False, 
          weight_decay=1e-4,
          hf_offline=False, 
          logger=None):
        
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.hf_offline = hf_offline
        self.diagnostics = dict()

        self.start_time = time.time()

        self.prompts = dict()
        self.logger = logger
        self.weight_decay = weight_decay
        self.linesearch = linesearch
        self.m = m
        self.smooth = smooth
        self.distribution = distribution

    def train_prompt_iteration(self, get_prompt, get_batch, 
          test_prompt_trajectories_list, test_trajectories_list, 
          eval_episodes, finetuning_hf_episodes, finetuning_hf_offline, env_name_list, 
          info, prompt_info,
          variant, env_list, iter_num=0,  
          no_prompt=False, group='test-finetune',
          ):

        self.logger.info('=' * 80)
        self.logger.info('finetune prompt at tasks: ')
        for i in env_name_list:
            self.logger.info(i)
        logs = dict()
        self.logger.info('start finetuning...')

        train_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            finetune_fns = finetuning_hf_episodes(info[env_name]['env_targets'][-1], info[env_name], variant, env_list[env_id], env_name)
            get_prompt_fns = get_prompt(test_prompt_trajectories_list[env_id], prompt_info[env_name], variant)
            get_batch_fns = get_batch(test_trajectories_list[env_id], info[env_name], variant)
            finetune_offline_fns = finetuning_hf_offline(get_batch_fns, variant, self.loss_fn)
            if not no_prompt:
                if not env_name in self.prompts.keys():
                  if 'vel' in env_name and variant['index'] is True:
                    prompt = flatten_prompt(get_prompt_fns(index=4), batch_size=1) # one prompt for the whole batch now
                  else:
                    prompt = flatten_prompt(get_prompt_fns(), batch_size=1) # one prompt for the whole batch now
                  self.prompts[env_name] = prompt
                else:
                  prompt = self.prompts[env_name]
            else:
                prompt = None

            self.model.train()
            taming_optimizer = RankSGD_Net(prompt, self.smooth, variant['finetune_lr'], 
                                           distribution=self.distribution, linesearch=self.linesearch)
            if self.hf_offline:
                oracle = NetworkOracle(finetune_offline_fns, self.model, self.m, self.m, weight_decay=self.weight_decay, offline=self.hf_offline)
            else:
                oracle = NetworkOracle(finetune_fns, self.model, self.m, self.m, weight_decay=self.weight_decay, offline=self.hf_offline)
            # finetune the model on the data for this task 
            finetune_losses = []
            for i in trange(variant['finetune_steps']):
                finetune_loss = self.train_step(
                    taming_optimizer=taming_optimizer,
                    oracle=oracle,
                    start=0)
                finetune_losses.append(finetune_loss)

            logs[f'finetuning-{env_name}-train_loss_mean'] = np.mean(finetune_losses)
            logs[f'finetuning-{env_name}-train_loss_std'] = np.std(finetune_losses)

            self.model.eval()
            prompt = taming_optimizer.best_ret()
            self.prompts[env_name] = prompt
            # need to sample eval_fn and prompt together 
            for eval_fn in eval_fns:
                outputs = eval_fn(self.model, prompt=prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        
        logs['time/finetuning'] = time.time() - train_start

        total_return_mean = []
        self.logger.info(f'Iteration {iter_num}')
        for k, v in logs.items():
            self.logger.info(f'{k}: {v}')
            if 'return_mean' in k:
                total_return_mean.append(float(v))
        self.logger.info(f'Total return mean is {np.mean(total_return_mean)}')
        logs[f'{group}-evaluation/Total_Return_Mean'] = np.mean(total_return_mean)

        return logs
    
    def train_step(self, taming_optimizer, oracle, start):
        with torch.no_grad():
            if self.linesearch is False:
                for j in range(1):
                    test_prompt = taming_optimizer.generate_query_parameter(self.m)
                    rank_info, loss = oracle(test_prompt, start)
                    taming_optimizer.update(rank_info)
            else:
                for j in range(2):
                    test_para = taming_optimizer.generate_query_parameter(self.m)
                    rank_info, loss = oracle(test_para, start)
                    taming_optimizer.update(rank_info)

        return loss

    def save_model(self, env_name, postfix, folder):
        model_name = '/prompt_model_' + env_name + postfix
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        for k, v in self.prompts.items():
            state_dict[k] = v

        torch.save(state_dict, folder+model_name)  # model save
        self.logger.info('model saved to ' + folder+model_name)