#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ppo.py
@Time    :   2023/07/14 11:23:57
@Author  :   Zhou Zihao 
@Version :   1.0
@Desc    :   None
'''

from .base import Base
import numpy as np
import torch
import torch.nn.functional as F
from .model import Critic_network
import torch.optim as optim
import torch.nn as nn

class PPO(Base):

    def __init__(self, 
                 model,
                 obs_space,
                 action_space,
                 device, 
                 save_path,
                 recurrent=False,
                 lr=0.001, 
                 max_grad_norm=0.5, 
                 adam_eps=1e-8,
                 clip_eps=0.2,
                 entropy_coef=0.001,
                 kickstarting_coef_initial=10.,
                 kickstarting_coef_decent=0.005,
                 kickstarting_coef_minimum=0.1,
                 iter_with_ks=3000,
                 value_loss_coef=.5,  
                 batch_size=128,
                 num_worker=4, 
                 epoch=3,
                ):
        # model
        super().__init__(model, obs_space, action_space, device, save_path, recurrent)

        # optimizer
        self.lr                = lr
        self.grad_clip         = max_grad_norm
        self.eps               = adam_eps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, eps=self.eps)
        
        # loss
        self.clip              = clip_eps
        self.entropy_coef      = entropy_coef
        self.iter_with_ks      = iter_with_ks
        self.ks_coef           = kickstarting_coef_initial
        self.ks_coef_minimum   = kickstarting_coef_minimum
        self.ks_coef_descent   = kickstarting_coef_decent
        self.value_loss_coef   = value_loss_coef
        self.meta_controller_value_loss_coef   = 1
        self.meta_controller_pi_loss_coef = 1
        
        # other settings 
        self.batch_size        = batch_size
        self.epochs            = epoch
        self.num_worker        = num_worker
        self.iter              = 0
        self.eplison = 1e-3
        
    def __call__(self, obs, mask, states):
        return self.model(obs, mask, states)
        
    def update_kickstarting_coef(self):
        self.iter += 1
        if self.ks_coef <= self.ks_coef_minimum:
            self.ks_coef = self.ks_coef_minimum
        else:
            self.ks_coef -= self.ks_coef_descent
        # if self.iter == self.iter_with_ks:
        #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, eps=self.eps)
        #     self.model.actor[-1].reset_parameters()

    def update_policy(self, buffer):
        losses = []
        torch.autograd.set_detect_anomaly(True)
        for _ in range(self.epochs):
            for batch in buffer.sample(self.batch_size, self.recurrent):
                obs_batch, _, action_batch, return_batch, advantage_batch, values_batch, mask, log_prob_batch, controller_prob_batch, meta_controller_prob_batch, meta_controller_value_batch, meta_controller_tensor_batch, _ = batch

                # get policy
                states = self.model.init_states(self.device, obs_batch.size()[1]) if self.recurrent else None
                pdf, value, _ = self.model(obs_batch, mask, states)

                # update
                entropy_loss = (pdf.entropy() * mask).mean()
                kickstarting_loss = -(pdf.logits * meta_controller_prob_batch).sum(dim=-1).mean()

                ratio = torch.exp(pdf.log_prob(action_batch) - log_prob_batch)
                surr1 = ratio * advantage_batch * mask
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = values_batch + torch.clamp(value - values_batch, -self.clip, self.clip)
                surr1 = ((value - return_batch)*mask).pow(2)
                surr2 = ((value_clipped - return_batch)*mask).pow(2)
                value_loss = torch.max(surr1, surr2).mean()
                
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss 
                if self.iter < self.iter_with_ks:
                    loss += self.ks_coef * kickstarting_loss
                

                # Update actor-critic
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Update logger
                losses.append([loss.item(), entropy_loss.item(), kickstarting_loss.item(), policy_loss.item(), value_loss.item()])
                
        if self.iter < self.iter_with_ks:
            # Update kickstarting coefficient
            self.update_kickstarting_coef()
        
        mean_losses = np.mean(losses, axis=0)
        return mean_losses
    
    def update_network(self, buffer):
        losses = []
        for _ in range(self.epochs):
            for batch in buffer.sample(self.batch_size, self.recurrent):
                obs_batch, _, action_batch, return_batch, advantage_batch, values_batch, mask, log_prob_batch, controller_prob_batch, meta_controller_prob_batch, meta_controller_value_batch, meta_controller_tensor_batch, _ = batch
                # get policy
                states = self.model.init_states(self.device, obs_batch.size()[1]) if self.recurrent else None
                pdf, value, _ = self.model(obs_batch, mask, states)

                # imitate q value
                value_dis_clipped = F.mse_loss(meta_controller_value_batch, values_batch)
                value_dis_clipped = torch.clamp(value_dis_clipped, -self.clip, self.clip)
                value_dis_loss = value_dis_clipped.mean()

                #imitate pi
                controller_logits = pdf.logits
                kl_div_per_sample = F.cross_entropy(controller_logits, meta_controller_tensor_batch, reduction='mean')
                average_kl_div = kl_div_per_sample.mean()

                loss = (self.meta_controller_value_loss_coef * value_dis_loss + self.meta_controller_pi_loss_coef * average_kl_div) * 5

                # Update actor-critic
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Update logger
                losses.append([loss.item(), value_dis_loss.item(), average_kl_div.item()])

        mean_losses = np.mean(losses, axis=0)
        return mean_losses



class Critic():
    def __init__(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space
        self.critic_network = Critic_network(self.obs_space, self.action_space)
        self.gamma = 0.9
        self.batch_size = 128
        self.learning_rate = 0.01
        self.epochs = 3
        self.optimizer = optim.Adam(self.critic_network.parameters(), self.learning_rate)
        self.grad_clip = 0.5
        self.max_value = 2
        self.coef = 0.5

    def __call__(self, state):
        clipped_output = torch.clamp(self.critic_network(state), min=-5, max=5)
        return clipped_output
    
    def update_policy(self, buffer, upp):
        for _ in range(self.epochs):
            losses = []
            for batch in buffer.sample(self.batch_size):
                obs_batch, next_obs_batch, _, return_batch, _, mask, _, _, _, _, meta_controller_value_batch, _, done = batch

                states = self.model.init_states(self.device, obs_batch.size()[1]) if self.recurrent else None
                pdf, value, _ = self.model(obs_batch, mask, states)

                controller_logits = pdf.logits
                kl_div_per_sample = F.cross_entropy(controller_logits, meta_controller_value_batch, reduction='mean')

                v_next = self.critic_network(next_obs_batch)
                v_next = torch.minimum(v_next, torch.tensor(self.max_value, device=v_next.device))
                return_batch = return_batch.unsqueeze(1)
                max_q_value = torch.max(v_next).item()
                max_q_values = torch.full_like(return_batch, max_q_value, requires_grad=True)
                done = done.squeeze()
                one_minus_done = 1 - done
                one_minus_done = one_minus_done.unsqueeze(1)
                predicted_values = torch.minimum(torch.maximum(return_batch + self.gamma * max_q_values * one_minus_done - self.coef *kl_div_per_sample, torch.tensor(0, device=v_next.device)), torch.tensor(self.max_value, device=v_next.device))
                meta_controller_value_batch =  torch.minimum(torch.maximum(meta_controller_value_batch, torch.tensor(-self.max_value, device=meta_controller_value_batch.device)), torch.tensor(self.max_value, device=meta_controller_value_batch.device))
                loss = nn.functional.mse_loss(predicted_values, meta_controller_value_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.grad_clip)
                self.optimizer.step()
                losses.append(loss.item())
        mean_losses = np.mean(losses, axis=0)
        return mean_losses
    
