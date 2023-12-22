# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
from wandb import env
from .prompt_utils import flatten_prompt
from .fl_utils import apply_mask_grad, apply_mask_model, load_original_parameters, apply_final_mask, cosine_annealing, mask_grow, mask_death, parameters_to_gradvector
import copy
from tqdm import tqdm, trange


class PromptSequenceTrainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn,
                 scheduler=None, eval_fns=None, get_prompt=None, get_prompt_batch=None, 
                 logger=None, variant=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.env_model_dict = dict()
        self.get_prompt = get_prompt
        self.prompt = self.get_prompt() # sample prompt data when initialization
        self.get_prompt_batch = get_prompt_batch

        self.eta_min = variant['eta_min']
        self.eta_max = variant['eta_max']
        self.sparsity = variant['sparsity']

        self.start_time = time.time()
        self.logger = logger


    def pure_train_iteration_mix(self, num_steps, no_prompt=False, masks=None, env_name=None):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for i in range(num_steps):
            train_loss, masks = self.train_step_mix(no_prompt, env_name, masks)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs, masks
    
    def train_step_mix(self, no_prompt=False, index=None, masks=None):
        prompt, batch = self.get_prompt_batch(index=index)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name = batch
        action_target = torch.clone(actions)

        original_param = copy.deepcopy(self.model.state_dict())
        param_after_mask = apply_mask_model(self.model, masks)
        self.model.load_state_dict(param_after_mask)

        if no_prompt:
            state_preds, action_preds, reward_preds = self.model.forward(
                env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=prompt
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        apply_mask_grad(self.model, masks)
        self.optimizer.step()

        load_original_parameters(self.model, original_param, masks)

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()


        return loss.detach().cpu().item(), masks

    def update_gradient(self, num_steps, no_prompt=False, masks=None, env_name=None):

        prompt, batch = self.get_prompt_batch(index=env_name)
        states, actions, rewards, dones, rtg, timesteps, attention_mask, env_name_ = batch
        action_target = torch.clone(actions)

        original_param = copy.deepcopy(self.model.state_dict())
        param_after_mask = apply_mask_model(self.model, masks)
        self.model.load_state_dict(param_after_mask)
                
        self.model.train()

        if no_prompt:
            state_preds, action_preds, reward_preds = self.model.forward(
                env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=None
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                env_name, states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask, prompt=prompt
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        apply_mask_grad(self.model, masks)

        gradient = parameters_to_gradvector(self.model)

        # load_original_parameters(self.model, original_param, masks)
        self.model.load_state_dict(original_param)

        return gradient

    def eval_iteration_multienv(self, env_masks, get_prompt, prompt_trajectories_list, eval_episodes, env_name_list, info, prompt_info,
                                variant, env_list, iter_num=0, print_logs=False, no_prompt=False, group='test', prompts=None,
                                ):
        
        self.logger.log('=' * 80)
        self.logger.log('evaluate at tasks: ')
        for i in env_name_list:
            self.logger.log(i)
        logs = dict()
        self.logger.log('start evaluating...')
        self.model.eval()

        model = copy.deepcopy(self.model)
        params = apply_final_mask(model, env_masks, 1)
        model.load_state_dict(params)

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            
            # need to sample eval_fn and prompt together 
            self.logger.log(f'Evaluate at task: {env_name}')
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            self.get_prompt = get_prompt(prompt_trajectories_list[env_id], prompt_info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(index=0), batch_size=1)
                if prompts is not None:
                    self.prompt = prompts[env_name]
            else:
                self.prompt = None
            for eval_fn in self.eval_fns:
                # print('env_name : ', env_list[env_id])
                outputs = eval_fn(model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        total_return_mean = {}
        self.logger.record_tabular('Iteration', iter_num)
        for k, v in logs.items():
            self.logger.record_tabular(k, float(v))
            if 'return_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if env not in total_return_mean.keys():
                    total_return_mean[env] = float(v)
                elif total_return_mean[env] < float(v):
                    total_return_mean[env] = float(v)
        total_mean = []
        for k, v in total_return_mean.items():
            self.logger.record_tabular(k, float(v))
            total_mean.append(v)
        self.logger.record_tabular('Total return mean', np.mean(total_mean))
        self.logger.dump_tabular()
        logs[f'Total_Return_Mean'] = np.mean(total_mean)

        return logs

    def eval_iteration_metaworld(self, env_masks, get_prompt, prompt_trajectories_list, eval_episodes, env_name_list, info, prompt_info,
                                variant, env_list, iter_num=0, no_prompt=False, group='test', seperate_test=False, merge_thres=1.
                                ):
        
        self.logger.log('=' * 80)
        self.logger.log('evaluate at tasks: ')
        for i in env_name_list:
            self.logger.log(i)
        logs = dict()
        self.logger.log('start evaluating...')
        self.model.eval()

        model = copy.deepcopy(self.model)
        params = apply_final_mask(model, env_masks, merge_thres)
        model.load_state_dict(params)

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):
            
            # need to sample eval_fn and prompt together 
            self.logger.log(f'Evaluate at task: {env_name}')
            self.eval_fns = [eval_episodes(tar, info[env_name], variant, env_list[env_id], env_name) for tar in info[env_name]['env_targets']]
            self.get_prompt = get_prompt(prompt_trajectories_list[env_id], prompt_info[env_name], variant)
            if not no_prompt:
                self.prompt = flatten_prompt(self.get_prompt(index=0), batch_size=1)
            else:
                self.prompt = None
            
            if seperate_test is True:
                masks = env_masks[env_name]
                model = copy.deepcopy(self.model)
                param_after_mask = apply_mask_model(model, masks)
                model.load_state_dict(param_after_mask)

            
            for eval_fn in self.eval_fns:
                outputs = eval_fn(model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        total_return_mean = {}
        total_success_mean = {}
        self.logger.record_tabular('Iteration', iter_num)
        for k, v in logs.items():
            self.logger.record_tabular(k, float(v))
            if 'return_mean' in k:
                env = k.split('/')[1].split('_')[0]
                if env not in total_return_mean.keys():
                    total_return_mean[env] = float(v)
                elif total_return_mean[env] < float(v):
                    total_return_mean[env] = float(v)
            if 'success_mean' in k:
                if env not in total_success_mean.keys():
                    total_success_mean[env] = float(v)
                elif total_success_mean[env] < float(v):
                    total_success_mean[env] = float(v)
        self.logger.dump_tabular()
        
        total_mean = []
        total_success = []
        for k, v in total_return_mean.items():
            # self.logger.record_tabular(f'{group}-{k}-Return', float(v))
            self.logger.record_tabular(f'{group}-{k}-Success', float(total_success_mean[k]))
            total_mean.append(v)
            total_success.append(total_success_mean[k])
        # self.logger.record_tabular('Total return mean', np.mean(total_mean))
        self.logger.record_tabular(f'{group}-Total-success-mean', np.mean(total_success))
        self.logger.dump_tabular()
        logs[f'{group}-Total-Return-Mean'] = np.mean(total_mean)
        logs[f'{group}-Total-Success-Mean'] = np.mean(total_success)

        return logs

 
    def save_model(self, env_name, postfix, folder, env_masks):

        model_name = '/prompt_model_' + postfix
        saved = {
            'model': self.model.state_dict(),
            'env_masks': env_masks,
        }
        torch.save(saved, folder+model_name)  # model save
        self.logger.log('model saved to ' + folder+model_name)
