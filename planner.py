#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/05/16 09:12:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


import os, requests
from typing import Any
from mediator import *
from utils import global_param

import numpy as np


from abc import ABC, abstractmethod

class Base_Planner(ABC):
    """The base class for Planner."""

    def __init__(self, offline=False, soft=False, prefix='', LLM_noise = "False", LLM_noise_ratio = 0.2, pi_meta_controller_noise = "gaussian", noise_ratio = 0.2):
        super().__init__()
        self.offline = offline
        self.soft = soft
        self.prompt_prefix = prefix
        self.plans_dict = {}
        self.mediator = None
        self.pi_meta_controller_noise = pi_meta_controller_noise
        self.noise_ratio = noise_ratio
        
        self.dialogue_system = ''               
        self.dialogue_user = ''
        self.dialogue_logger = ''     
        self.show_dialogue = False
        self.choices = {}
        print(f"offline: {offline}")
        if not offline:
            self.llm_model = 'meta_controller'
            self.llm_url = 'http://localhost:8000/v1/chat/completions'
            # self.llm_model = "chatglm_Turbo"
            # self.llm_url = 'http://10.109.116.3:6000/chat'
            self.plans_dict = {}
            if self.llm_model == 'meta_controller':
                self.init_llm()
                print("init LLM")
        
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)
        self.mediator.reset()
        # if not self.offline:
        #     self.online_planning("reset")
        
    def init_llm(self):
        self.dialogue_system += self.prompt_prefix

        ## set system part
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                headers = {'Content-Type': 'application/json'}
                
                data = {'model': self.llm_model, "messages":[{"role": "system", "content": self.prompt_prefix}]}
                response = requests.post(self.llm_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    assert False, f"fail to initialize: status code {response.status_code}"                
                    
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to initialize: {e}")

    def query_codex(self, prompt_text):
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                #response =  openai.Completion.create(prompt_text)
                headers = {'Content-Type': 'application/json'}
                
                # print(f"user prompt:{prompt_text}")
                if self.llm_model == "chatglm_Turbo":
                    data = {'model': self.llm_model, "prompt":[{"role": "user", "content": self.prompt_prefix + prompt_text}]}     
                elif self.llm_model == 'meta_controller':
                    data = {'model': self.llm_model, "messages":[{"role": "user", "content": prompt_text}]}
                response = requests.post(self.llm_url, headers=headers, json=data)

                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    assert False, f"fail to query: status code {response.status_code}"
       
            except Exception as e:
                server_error_cnt += 1
                print(f"fail to query: {e}")

        try:
            choices = result['choices'][0]
            message = choices['message']
            content = message['content']
            # plan = re.search("Action[s]*\:\s*\{([\w\s\<\>\,]*)\}", content, re.I | re.M).group(1)
            return content
        except:
            print(f"LLM response invalid format: '{result}'.")
            return self.query_codex(prompt_text)   
        
    def plan(self, text, n_ask=10):
        if text in self.plans_dict.keys():
            plans, probs = self.plans_dict[text]
        else:
            print(f"new obs: {text}")
            plans = {}
            for _ in range(n_ask):
                plan = self.query_codex(text)
                if plan in plans.keys():
                    plans[plan] += 1/n_ask
                else:
                    plans[plan] = 1/n_ask
            
            plans, probs = list(plans.keys()), list(plans.values())

            if self.pi_meta_controller_noise == "gaussian":
                n = len(probs)
                for i in range(n):
                    noise = np.random.normal(0.5, 0.1)
                    noise = np.clip(noise, 0, 1)
                    probs[i] += (self.noise_ratio * noise).item()
            total = sum(probs)
            probs = [x / total for x in probs]
            print(probs)

            self.plans_dict[text] = (plans, probs)
            
            for k, v in self.plans_dict.items():
                print(f"{k}:{v}")

        return plans, probs
    
    def __call__(self, obs):
        # self.mediator.reset()
        text = self.mediator.RL2LLM(obs)
        try:
            l_choices = self.choices[text]
            s_choices = '['
            for i in l_choices:
                i = f"'{i}', "
                s_choices += i
            s_choices += ']'
            addition_string = f"Choose the next rational action from {s_choices}. No further explanation."
            
        except:
            addition_string = "The observation is corrupted. Choose a possible action from ['explore','pick up <key>', 'go to <door>', 'open <door>']"
        text += addition_string
        plans, probs = self.plan(text)
        self.dialogue_user = text + "\n" + str(plans) + "\n" + str(probs)
        if self.show_dialogue:
            print(self.dialogue_user)
        skill_list, probs = self.mediator.LLM2RL(plans, probs)
        
        return skill_list, probs
    
    

class SimpleDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix, pi_meta_controller_noise = "gaussian", noise_ratio = 0.2, LLM_noise = "False", LLM_noise_ratio = 0.2):
        super().__init__(offline, soft, prefix, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio, pi_meta_controller_noise = pi_meta_controller_noise, noise_ratio = noise_ratio)
        self.mediator = SimpleDoorKey_Mediator(soft, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>." : [["explore"], [1.0]],
                "Agent sees <door>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>."   : [["go to <key>, pick up <key>", "pick up <key>"], [0.98, 0.02]],
                "Agent sees <nothing>, holds <key>."     : [["explore", "go to <door>, open <door>", "explore, go to <door>, open <door>", "explore, go to <door>", "explore, open <door>", "go to <door>, pick up <handle>, use <key>"], [0.68, 0.22, 0.04, 0.02, 0.02, 0.02]],
                "Agent sees <door>, holds <key>."        : [["go to <door>, open <door> with <key>", "go to <door>, open <door>", "go to <key>, pick up <key>, go to <door>, open <door>", "explore, go to <door>"], [0.62, 0.3, 0.06, 0.02]],
                "Agent sees <key>, <door>, holds <nothing>." : [["go to <key>, pick up <key>, go to <door>, open <door>", "go to <key>, pick up <key>, open <door>", "pick up <key>, go to <door>, open <door>", "go to <key>, go to <door>, use <key>", "go to <key>, pick up <key>, explore"], [0.84, 0.08, 0.04, 0.02, 0.02]]
}     
        self.choices = {
                "Agent sees <nothing>, holds <nothing>." : ["explore"],
                "Agent sees <door>, holds <nothing>."  : ["explore"],
                "Agent sees <key>, holds <nothing>."   : ["go to <key>, pick up <key>", "pick up <key>"],
                "Agent sees <nothing>, holds <key>."     :["explore", "go to <door>, open <door>", "explore, go to <door>, open <door>", "explore, go to <door>", "explore, open <door>", "go to <door>, pick up <handle>, use <key>"],
                "Agent sees <door>, holds <key>."        : ["go to <door>, open <door> with <key>", "go to <door>, open <door>", "go to <key>, pick up <key>, go to <door>, open <door>", "explore, go to <door>"],
                "Agent sees <door>, <key>, holds <nothing>." : ["go to <key>, pick up <key>, go to <door>, open <door>", "go to <key>, pick up <key>, open <door>", "pick up <key>, go to <door>, open <door>", "go to <key>, go to <door>, use <key>", "go to <key>, pick up <key>, explore"], 
        }

class KeyInBox_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix, pi_meta_controller_noise = "gaussian", noise_ratio = 0.2, LLM_noise = "False", LLM_noise_ratio = 0.2):
        super().__init__(offline, soft, prefix, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio, pi_meta_controller_noise = pi_meta_controller_noise, noise_ratio = noise_ratio)
        self.mediator = KeyInBox_Mediator(soft, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio)
        self.choices = {
            "Agent sees <nothing>, holds <nothing>." : ["explore"],
            "Agent sees <door>, holds <nothing>."  : ["explore"], 
            "Agent sees <key>, holds <nothing>."   : ["go to <key> and pick up <key>", "pick up <key>"],
            "Agent sees <nothing>, holds <key>."     : ["explore", "go to <door>, open <door>", "explore, go to <door>, open <door>", "explore, go to <door>", "explore, open <door>", "go to <door>", "go to <box>", "go to <box> and toggle <box>", "toggle <box>"],
            "Agent sees <door>, holds <key>."        : ["go to <door>, open <door> with <key>", "go to <door>, open <door>", "go to <key>, pick up <key>, go to <door>, open <door>", "explore, go to <door>", "go to <box>", "go to <box> and toggle <box>", "toggle <box>"], 
            "Agent sees <key>, <door>, holds <nothing>." : ["go to <key>, pick up <key>, go to <door>, open <door>", "go to <key>, pick up <key>, open <door>", "pick up <key>, go to <door>, open <door>", "go to <key>, go to <door>, use <key>", "go to <key>, pick up <key>, explore", "go to <box>", "go to <box> and toggle <box>", "toggle <box>"],
             
            "Agent sees <box>, holds <nothing>." : ["go to <box> and toggle <box>", "go to <box>", "go to <key>", "go to <door>", "go to <door> and open <door>", "explore", "go to <box>, toggle <box>, pick up <key>", "go to <box>, toggle <box>, pick up <key>, go to <door>, open <door>", "toggle <box>, pick up <key> and go to <door>"],
            "Agent sees <box>, holds <key>." : ["explore", "go to <door> and open <door>", "go to <box>", "go to <box> and toggle <box>", "go to <key> and pick up <key>"],
            "Agent sees <box>, <door>, holds <nothing>." : ["go to <box> and toggle <box>", "go to <box>", "go to <key>", "go to <door>", "go to <door> and open <door>", "explore", "go to <box>, toggle <box>, pick up <key>", "go to <box>, toggle <box>, pick up <key>, go to <door>, open <door>", "toggle <box>, pick up <key> and go to <door>"],
            "Agent sees <box>, <door>, <key>, holds <nothing>." : ["go to <key> and pick up <key>", "go to <box>", "go to <key>", "go to <door>", "go to <door> and open <door>", "explore", "go to <box>, toggle <box>, pick up <key>", "go to <box>, toggle <box>, pick up <key>, go to <door>, open <door>", "toggle <box>, pick up <key> and go to <door>"],
        }
        #self.choices = {
        #     "Agent sees <nothing>, holds <nothing>." : ["explore"],
        #     "Agent sees <door>, holds <nothing>."  : ["explore"], 
        #     "Agent sees <key>, holds <nothing>."   : ["go to <key>, pick up <key>"],
        #     "Agent sees <nothing>, holds <key>."     : ["explore"],
        #     "Agent sees <door>, holds <key>."        : ["go to <door>, open <door>"], 
        #     "Agent sees <key>, <door>, holds <nothing>." : ["go to <key>, pick up <key>"],
             
        #     "Agent sees <box>, holds <nothing>." : ["go to <box>, toggle <box>"],
        #     "Agent sees <box>, holds <key>." : ["explore"],
        #     "Agent sees <box>, <door>, holds <nothing>." : ["go to <box>, toggle <box>"],
        #     "Agent sees <box>, <door>, <key>, holds <nothing>." : ["go to <key>, pick up <key>"],
        # }
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)


    def __call__(self, obs):
        # self.mediator.reset()
        text = self.mediator.RL2LLM(obs)
        try:
            l_choices = self.choices[text]
            s_choices = '['
            for i in l_choices:
                i = f"'{i}', "
                s_choices += i
            s_choices += ']'
            addition_string = f"Choose the next rational action from {s_choices}. No further explanation."
            
        except:
            addition_string = "The observation is corrupted. Choose a possible action from ['explore','pick up <key>', 'go to <door>', 'open <door>', 'go to <box>', 'toggle <box>']"
            
        text += addition_string
        plans, probs = self.plan(text)
        # print(f"plan is {plans}\n=================================================\n")
        self.dialogue_user = text + "\n" + str(plans) + "\n" + str(probs)
        if self.show_dialogue:
            print(self.dialogue_user)
        skill_list, probs = self.mediator.LLM2RL(plans, probs)
        
        return skill_list, probs

class ColoredDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = ColoredDoorKey_Mediator(soft)

        self.choices = {
            "Agent sees <nothing>, holds <nothing>."       : ["explore"],
            "Agent sees <nothing>, holds <color1 key>."    : ["explore","go to east"],
            "Agent sees <color1 key>, holds <nothing>."    : ["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],
            "Agent sees <color1 door>, holds <nothing>."   : ["explore"],
            "Agent sees <color1 door>, holds <color1 key>.": ["go to <color1 door>, open <color1 door>","open <color1 door>"],
            "Agent sees <color1 door>, holds <color2 key>.": ["explore", "go to <color2 key>"],
            "Agent sees <color1 key>, holds <color2 key>.": ["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],
            "Agent sees <color1 key>, <color2 key>, holds <nothing>.": ["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],
            "Agent sees <color1 key>, <color2 door>, holds <nothing>.": ["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],
            "Agent sees <color1 key>, <color1 door>, holds <nothing>.": ["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],
            "Agent sees <color1 key>, <color1 door>, holds <color2 key>.": ["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],
            "Agent sees <color1 key>, <color2 door>, holds <color2 key>.": ["drop <color2 key>, go to <color1 key>, pick up <color1 key>", "go to <color2 door>, open <color2 door>"],
            "Agent sees <color1 key>, <color2 key>, <color2 door>, holds <nothing>.": ["go to <color2 key>, pick up <color2 key>","pick up <color2 key>","go to <color1 key>, pick up <color1 key>"],
            "Agent sees <color1 key>, <color2 key>, <color1 door>, holds <nothing>.": ["go to <color1 key>, pick up <color1 key>"," pick up <color1 key>"],
        }
        
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>."       : [["explore"],[1]],
                "Agent sees <nothing>, holds <color1 key>."    : [["explore","go to east"], [0.94,0.06]],
                "Agent sees <color1 key>, holds <nothing>."    : [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.87,0.13]],
                "Agent sees <color1 door>, holds <nothing>."   : [["explore"],[1.0]],
                "Agent sees <color1 door>, holds <color1 key>.": [["go to <color1 door>, open <color1 door>","open <color1 door>"],[0.72,0.28]],
                "Agent sees <color1 door>, holds <color2 key>.": [["explore", "go to <color2 key>"],[0.98,0.02]],
                "Agent sees <color1 key>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],[0.87,0.13]],
                "Agent sees <color1 key>, <color2 key>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.81,0.19]],
                "Agent sees <color1 key>, <color2 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.73,0.27]],
                "Agent sees <color1 key>, <color1 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>","pick up <color1 key>"],[0.84,0.16]],
                "Agent sees <color1 key>, <color1 door>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>","drop <color2 key>, pick up <color1 key>"],[0.79,0.21]],
                "Agent sees <color1 key>, <color2 door>, holds <color2 key>.": [["drop <color2 key>, go to <color1 key>, pick up <color1 key>", "go to <color2 door>, open <color2 door>"],[0.71,0.29]],
                "Agent sees <color1 key>, <color2 key>, <color2 door>, holds <nothing>.": [["go to <color2 key>, pick up <color2 key>","pick up <color2 key>","go to <color1 key>, pick up <color1 key>"],[0.72,0.24,0.04]],
                "Agent sees <color1 key>, <color2 key>, <color1 door>, holds <nothing>.": [["go to <color1 key>, pick up <color1 key>"," pick up <color1 key>"],[0.94,0.06]],
            }
        
    def plan(self, text):
        pattern= r'\b(blue|green|grey|purple|red|yellow)\b'
        color_words = re.findall(pattern, text)

        words = list(set(color_words))
        words.sort(key=color_words.index)
        color_words = words
        color_index =['color1','color2']
        if color_words != []:
            for i in range(len(color_words)):
                text = text.replace(color_words[i], color_index[i])

        plans, probs = super().plan(text)

        plans = str(plans)
        for i in range(len(color_words)):
            plans = plans.replace(color_index[i], color_words[i])
        plans = eval(plans)

        return plans, probs


class TwoDoor_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = TwoDoor_Mediator(soft)
        self.choices = {
            "Agent sees <nothing>, holds <nothing>." : ["explore"],
            "Agent sees <door1>, holds <nothing>."  : ["explore"], 
            "Agent sees <key>, holds <nothing>."   : ["go to <key>, pick up <key>"],
            "Agent sees <nothing>, holds <key>."     : ["explore"],
            "Agent sees <door1>, holds <key>."        : ["go to <door1>, open <door1>"], 
            "Agent sees <key>, <door1>, holds <nothing>." : ["go to <key>, pick up <key>"], 
            "Agent sees <door1>, <door2>, holds <nothing>."  : ["explore"],
            "Agent sees <key>, <door1>, <door2>, holds <nothing>.": ["go to <key>, pick up <key>"], 
            "Agent sees <door1>, <door2>, holds <key>.": ["go to <door1>, open <door1>", "go to <door2>, open <door2>"]
        }
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>." : [["explore"], [1.0]],
                "Agent sees <door1>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>."   : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <nothing>, holds <key>."     : [["explore"], [1.0]],
                "Agent sees <door1>, holds <key>."        : [["go to <door1>, open <door1>"], [1.0]],
                "Agent sees <key>, <door1>, holds <nothing>." : [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <nothing>."  : [["explore"], [1.0]],
                "Agent sees <key>, <door1>, <door2>, holds <nothing>.": [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <key>.": [["go to <door1>, open <door1>", "go to <door2>, open <door2>"], [0.5, 0.5]],
            }  
                                                            
class RandomBoxKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix, pi_meta_controller_noise = "gaussian", noise_ratio = 0.2, LLM_noise = "False", LLM_noise_ratio = 0.2):
        super().__init__(offline, soft, prefix, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio, pi_meta_controller_noise = pi_meta_controller_noise, noise_ratio = noise_ratio)
        self.mediator = RandomBoxKey_Mediator(soft, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio)
        self.choices = {
            "Agent sees <nothing>, holds <nothing>." : ["explore"],
            "Agent sees <door>, holds <nothing>."  : ["explore"], 
            "Agent sees <key>, holds <nothing>."   : ["go to <key> and pick up <key>", "pick up <key>"],
            "Agent sees <nothing>, holds <key>."     : ["explore", "go to <door>, open <door>", "explore, go to <door>, open <door>", "explore, go to <door>", "explore, open <door>", "go to <door>", "go to <box>", "go to <box> and toggle <box>", "toggle <box>"],
            "Agent sees <door>, holds <key>."        : ["go to <door>, open <door> with <key>", "go to <door>, open <door>", "go to <key>, pick up <key>, go to <door>, open <door>", "explore, go to <door>", "go to <box>", "go to <box> and toggle <box>", "toggle <box>"], 
            "Agent sees <key>, <door>, holds <nothing>." : ["go to <key>, pick up <key>, go to <door>, open <door>", "go to <key>, pick up <key>, open <door>", "pick up <key>, go to <door>, open <door>", "go to <key>, go to <door>, use <key>", "go to <key>, pick up <key>, explore", "go to <box>", "go to <box> and toggle <box>", "toggle <box>"],
             
            "Agent sees <box>, holds <nothing>." : ["go to <box> and toggle <box>", "go to <box>", "go to <key>", "go to <door>", "go to <door> and open <door>", "explore", "go to <box>, toggle <box>, pick up <key>", "go to <box>, toggle <box>, pick up <key>, go to <door>, open <door>", "toggle <box>, pick up <key> and go to <door>"],
            "Agent sees <box>, holds <key>." : ["explore", "go to <door> and open <door>", "go to <box>", "go to <box> and toggle <box>", "go to <key> and pick up <key>"],
            "Agent sees <box>, <door>, holds <nothing>." : ["go to <box> and toggle <box>", "go to <box>", "go to <key>", "go to <door>", "go to <door> and open <door>", "explore", "go to <box>, toggle <box>, pick up <key>", "go to <box>, toggle <box>, pick up <key>, go to <door>, open <door>", "toggle <box>, pick up <key> and go to <door>"],
            "Agent sees <box>, <door>, <key>, holds <nothing>." : ["go to <key> and pick up <key>", "go to <box>", "go to <key>", "go to <door>", "go to <door> and open <door>", "explore", "go to <box>, toggle <box>, pick up <key>", "go to <box>, toggle <box>, pick up <key>, go to <door>, open <door>", "toggle <box>, pick up <key> and go to <door>"],
        }
    
    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        ## reset dialogue
        if self.show_dialogue:
            print(self.dialogue_system)


    def __call__(self, obs):
        # self.mediator.reset()
        text = self.mediator.RL2LLM(obs)
        try:
            l_choices = self.choices[text]
            s_choices = '['
            for i in l_choices:
                i = f"'{i}', "
                s_choices += i
            s_choices += ']'
            addition_string = f"Choose the next rational action from {s_choices}. No further explanation."
            
        except:
            addition_string = "The observation is corrupted. Choose a possible action from ['explore','pick up <key>', 'go to <door>', 'open <door>', 'go to <box>', 'toggle <box>']"
            
        text += addition_string
        plans, probs = self.plan(text)
        # print(f"plan is {plans}\n=================================================\n")
        self.dialogue_user = text + "\n" + str(plans) + "\n" + str(probs)
        if self.show_dialogue:
            print(self.dialogue_user)
        skill_list, probs = self.mediator.LLM2RL(plans, probs)
        
        return skill_list, probs      


def Planner(task, offline=True, soft=False, prefix='', LLM_noise = "False", LLM_noise_ratio = 0.2, pi_meta_controller_noise = "gaussian", noise_ratio = 0.2):
    if task.lower() == "simpledoorkey":
        planner = SimpleDoorKey_Planner(offline, soft, prefix, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio, pi_meta_controller_noise = pi_meta_controller_noise, noise_ratio = noise_ratio)
    elif task.lower() == "lavadoorkey":
        planner = SimpleDoorKey_Planner(offline, soft, prefix)
    elif task.lower() == "coloreddoorkey":
        planner = ColoredDoorKey_Planner(offline, soft, prefix)
    elif task.lower() == "twodoor":
        planner = TwoDoor_Planner(offline, soft, prefix)
    elif task.lower() == 'keyinbox' :
        planner = KeyInBox_Planner(offline, soft, prefix)
    elif task.lower() == 'randomboxkey' :
        planner = RandomBoxKey_Planner(offline, soft, prefix,LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio, pi_meta_controller_noise = pi_meta_controller_noise, noise_ratio = noise_ratio)
    return planner
                                                            
                                                            
