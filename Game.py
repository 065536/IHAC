#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Game.py
@Time    :   2023/07/14 11:06:59
@Author  :   Zhou Zihao
@Version :   1.0
@Desc    :   None
'''

import os, json, sys
import gymnasium as gym
import numpy as np
import torch
import cv2
import time

import env
import algos
import skill
import utils
import cv2
from meta_controller import MetaController
import copy
from algos import *
import torch.nn.functional as F

prefix = os.getcwd()
task_info_json = os.path.join(prefix, "prompt/task_info.json")

class Game:
    def __init__(self, args, training=True):
        # init seed
        self.seed = args.seed
        self.setup_seed(args.seed)
        # The discount factor of guidance
        self.LLM_noise = args.LLM_noise
        self.LLM_noise_ratio = args.LLM_noise_ratio
        self.pi_meta_controller_noise = args.pi_meta_controller_noise
        self.pi_meta_controller_noise_ratio = args.pi_meta_controller_noise_ratio
        self.meta_controller_coef = 1

        # init env
        self.load_task_info(args.task, args.frame_stack, args.offline_planner, args.soft_planner)

        # init logger
        self.logger = utils.create_logger(args, training)
        
        # init policy
        if args.loaddir:
            model_dir = os.path.join(args.logdir, args.policy, args.task, args.loaddir, args.loadmodel)
            policy = torch.load(model_dir)
        else:
            policy = None
        self.device = args.device
        self.batch_size = args.batch_size
        self.recurrent = args.recurrent
        # self.controller_policy = policy
        self.controller_policy = algos.PPO(policy, 
                                        self.obs_space,
                                        self.action_space,
                                        self.device, 
                                        self.logger.dir, 
                                        batch_size=self.batch_size, 
                                        recurrent=self.recurrent)
        
        self.meta_controller_value_network = algos.Critic(self.obs_space, self.action_space)
        self.meta_controller_value_network.critic_network.to(self.device)

        # init buffer
        self.gamma = args.gamma
        self.lam = args.lam
        self.buffer = algos.Buffer(self.gamma, self.lam, self.device)

        # other settings
        self.n_itr = args.n_itr
        self.traj_per_itr = args.traj_per_itr
        self.num_eval = args.num_eval
        self.eval_interval = args.eval_interval
        self.save_interval = args.save_interval
        self.total_steps = 0
                
        
    def setup_seed(self, seed):
        # setup seed for Numpy, Torch and LLM, not for env
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        
    def load_task_info(self, task, frame_stack, offline, soft):
        print(f"[INFO]: resetting the task: {task}")
        with open(task_info_json, 'r') as f:
            task_info = json.load(f)
        task = task.lower()
        
        env_fn = utils.make_env_fn(task_info[task]['configurations'], 
                                   render_mode="rgb_array", 
                                   frame_stack = frame_stack)
        self.env = utils.WrapEnv(env_fn)
        self.obs_space = utils.get_obss_preprocessor(self.env.observation_space)[0]
        self.action_space = self.env.action_space.n
        self.max_ep_len = self.env.max_steps

        prefix = task_info[task]['description'] + task_info[task]['example']
        self.meta_controller_policy = MetaController(task, offline, soft, prefix, self.action_space, self.env.agent_view_size, self.LLM_noise, self.LLM_noise_ratio, pi_meta_controller_noise = self.pi_meta_controller_noise, noise_ratio = self.pi_meta_controller_noise_ratio)

            
    def train(self):
        start_time = time.time()
        for itr in range(self.n_itr):
            self.itr = itr
            print("********** Iteration {} ************".format(itr))
            print("time elapsed: {:.2f} s".format(time.time() - start_time))

            ## collecting ##
            sample_start = time.time()
            self.buffer.clear()
            n_traj = self.traj_per_itr
            for _ in range(n_traj):
                self.collect()
            while len(self.buffer) < self.batch_size * 2:
                self.collect()
                n_traj += 1
            total_steps = len(self.buffer)    
            samp_time = time.time() - sample_start
            print("{:.2f} s to collect {:6n} timesteps | {:3.2f}sample/s.".format(samp_time, total_steps, (total_steps)/samp_time))
            self.total_steps += total_steps

            ## training ##
            optimizer_start = time.time()
            if itr < self.n_itr//10:
                # imitation update
                mean_meta_controller_loss = self.meta_controller_value_network.update_policy(self.buffer, self.max_ep_len)
                mean_losses = self.controller_policy.update_network(self.buffer)
                opt_time = time.time() - optimizer_start
            else:
                mean_meta_controller_loss = self.meta_controller_value_network.update_policy(self.buffer, self.max_ep_len)
                mean_losses = self.controller_policy.update_policy(self.buffer)
                opt_time = time.time() - optimizer_start
            
            try:
                print("{:.2f} s to optimizer| loss {:6.3f}".format(opt_time, mean_losses[0]))
            except:
                print(mean_losses)

            ## evaluate ##
            if itr % self.eval_interval == 0 and itr > 0:
                evaluate_start = time.time()
                eval_returns = []
                eval_lens = []
                eval_success = []
                for i in range(self.num_eval):
                    eval_outputs = self.evaluate(itr, record_frames=False)
                    eval_returns.append(eval_outputs[0])
                    eval_lens.append(eval_outputs[1])
                    eval_success.append(eval_outputs[2])
                eval_time = time.time() - evaluate_start
                print("{:.2f} s to evaluate.".format(eval_time))
            
            if itr % self.save_interval == 0 and itr > 0:
                self.controller_policy.save(str(itr))

            if itr % 10 == 1:
                self.meta_controller_coef = self.meta_controller_coef * 0.99
            
            ## log ##
            if self.logger is not None:
                avg_len = np.mean(self.buffer.ep_lens)
                avg_reward = np.mean(self.buffer.ep_returns)
                std_reward = np.std(self.buffer.ep_returns)
                success_rate = sum(i > 0 for i in self.buffer.ep_returns) / n_traj
                sys.stdout.write("-" * 49 + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Timesteps', self.total_steps) + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Return (train)', round(avg_reward,2)) + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Episode Length (train)', round(avg_len,2)) + "\n")
                sys.stdout.write("| %25s | %15s |" % ('Success Rate (train)', round(success_rate,2)) + "\n")
                if itr % self.eval_interval == 0 and itr > 0:
                    avg_eval_reward = np.mean(eval_returns)
                    avg_eval_len = np.mean(eval_lens)
                    eval_success_rate = np.sum(eval_success) / self.num_eval
                    sys.stdout.write("| %25s | %15s |" % ('Return (eval)', round(avg_eval_reward,2)) + "\n")
                    sys.stdout.write("| %25s | %15s |" % ('Episode Length (eval) ', round(avg_eval_len,2)) + "\n")
                    sys.stdout.write("| %25s | %15s |" % ('Success Rate (eval) ', round(eval_success_rate,2)) + "\n")
                    self.logger.add_scalar("Test/Return", avg_eval_reward, itr)
                    self.logger.add_scalar("Test/Eplen", avg_eval_len, itr)
                    self.logger.add_scalar("Test/Success Rate", eval_success_rate, itr)
                sys.stdout.write("-" * 49 + "\n")
                sys.stdout.flush()

                self.logger.add_scalar("Train/Return Mean", avg_reward, itr)
                self.logger.add_scalar("Train/Return Std", std_reward, itr)
                self.logger.add_scalar("Train/Eplen", avg_len, itr)
                self.logger.add_scalar("Train/Success Rate", success_rate, itr)
                self.logger.add_scalar("Train/Loss", mean_losses[0], itr)
                self.logger.add_scalar("Train/q meta_controller loss", mean_meta_controller_loss, itr)

                if itr < self.n_itr // 10:
                    self.logger.add_scalar("Train/value_dis_loss", mean_losses[1], itr)
                    self.logger.add_scalar("Train/average_kl_div", mean_losses[2], itr)
                else:
                    self.logger.add_scalar("Train/Mean Entropy", mean_losses[1], itr)
                    self.logger.add_scalar("Train/Kickstarting Loss", mean_losses[2], itr)
                    self.logger.add_scalar("Train/Policy Loss", mean_losses[3], itr)
                    self.logger.add_scalar("Train/Value Loss", mean_losses[4], itr)
                    self.logger.add_scalar("Train/Kickstarting Coef", self.controller_policy.ks_coef, itr)
                                
        self.controller_policy.save()


    def collect(self):
        '''
        collect episodic data.
        ''' 
        with torch.no_grad():
            obs = self.env.reset()
            done = False 
            ep_len = 0
            
            # reset controller policy
            mask = torch.FloatTensor([1]).to(self.device) # not done until episode ends
            states = self.controller_policy.model.init_states(self.device) if self.recurrent else None
            
            # reset meta_controller policy
            self.meta_controller_policy.reset()

            while not done and ep_len < self.max_ep_len:
                # get action from controller policy
                dist, value, states = self.controller_policy(torch.Tensor(obs).to(self.device),
                                                          mask, states)
                controller_logits = dist.logits
                controller_logits +=  1e-8
                controller_log_prob = torch.softmax(controller_logits, dim=-1)
                controller_log_prob = torch.log(controller_log_prob)

                # get action from meta_controller policy
                meta_controller_probs = self.meta_controller_policy(obs[0])
                meta_controller_tensor = torch.tensor(meta_controller_probs, device=self.device)
                meta_controller_logits = torch.log(meta_controller_tensor + 1e-8)

                if self.meta_controller_coef > 1e-5:
                    guide_logits = controller_logits + self.meta_controller_coef * meta_controller_logits
                    probs = F.softmax(guide_logits.squeeze(), dim=-1)
                    action = torch.multinomial(probs, 1)
                else:
                    action = dist.sample()
                log_probs = dist.log_prob(action)

                action = action.to("cpu").numpy()

                
                meta_controller_probs_new = torch.softmax(meta_controller_logits, dim=0)
                
                


                
                # interact with env
                next_obs, reward, done, info = self.env.step(action)
                meta_controller_value = self.meta_controller_value_network(torch.Tensor(obs).to(self.device))
                #kl_div_per_sample = F.kl_div(controller_log_prob, meta_controller_probs_new, reduction='batchmean')
                #print(kl_div_per_sample.item())


                # store in buffer
                self.buffer.store(obs, 
                                  next_obs,
                                  action, 
                                  reward, 
                                  value.to("cpu").numpy(), 
                                  log_probs.to("cpu").numpy(), 
                                  controller_log_prob.to("cpu").numpy(),
                                  meta_controller_probs,
                                  meta_controller_value.to("cpu").numpy(), 
                                  meta_controller_probs_new.to("cpu").numpy(),
                                  done
                )
                obs = next_obs
                ep_len += 1
            if done:
                value = 0.
            else:
                value = self.controller_policy(torch.Tensor(obs).to(self.device), 
                                            mask, states)[1].to("cpu").item()
                
                
            self.buffer.finish_path(last_val=value)
        
        
    def evaluate(self, itr=None, seed=None, record_frames=True, deterministic=False, meta_controller_policy=False):
        with torch.no_grad():
            # init env
            seed = seed if seed else np.random.randint(1000000)
            obs = self.env.reset(seed)
            done = False 
            ep_len = 0
            ep_return = 0.

            if meta_controller_policy:
                # init meta_controller policy
                self.meta_controller_policy.reset()
            else:
                # init controller policy
                mask = torch.Tensor([1.]).to(self.device) # not done until episode ends
                states = self.controller_policy.model.init_states(self.device) if self.recurrent else None

            # init vedio directory
            if record_frames:
                img_array = []
                img = self.env.get_mask_render()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_array.append(img)
                
                dir_name = 'meta_controller video' if meta_controller_policy else 'video'
                dir_path = os.path.join(self.logger.dir, dir_name)
                try:
                    os.makedirs(dir_path)
                except OSError:
                    pass

            while not done and ep_len < self.max_ep_len:
                if meta_controller_policy:
                    # get action from meta_controller policy
                    probs = self.meta_controller_policy(obs[0])
                    if deterministic:
                        action = np.argmax(probs)
                    else:
                        action = np.random.choice(self.action_space, p=probs)
                else:
                    # get action from controller policy
                    dist, value, states = self.controller_policy(torch.Tensor(obs).to(self.device), mask, states)
                    if deterministic:
                        action = torch.argmax(dist.probs).unsqueeze(0).to("cpu").numpy()
                    else:
                        action = dist.sample().to("cpu").numpy()

                # interact with env
                obs, reward, done, info = self.env.step(action)
                ep_return += float(reward)
                ep_len += 1
                
                if record_frames:
                    img = self.env.get_mask_render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_array.append(img)
                
            ep_success = 1 if ep_return > 0 else 0

            # save vedio
            if record_frames:
                height, width, layers = img.shape
                size = (width,height)
                video_name = "%s-%s.avi"%(itr, seed) if itr else "%s.avi"%seed
                video_path = os.path.join(dir_path, video_name)
                out = cv2.VideoWriter(video_path, 
                                      fourcc=cv2.VideoWriter_fourcc(*'DIVX'), 
                                      fps=3, 
                                      frameSize=size)

                for img in img_array:
                    out.write(img)
                out.release()
                
            return ep_return, ep_len, ep_success
    
        
if __name__ == '__main__':
    pass


