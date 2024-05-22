#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mediator.py
@Time    :   2023/05/16 10:22:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import numpy as np
import re
import copy
from abc import ABC, abstractmethod
@staticmethod
def get_minigrid_words():
    colors = ["red", "green", "blue", "yellow", "purple", "grey"]
    objects = [
        "unseen",
        "empty",
        "wall",
        "floor",
        "box",
        "key",
        "ball",
        "door",
        "goal",
        "agent",
        "lava",
    ]

    verbs = [
        "pick",
        "avoid",
        "get",
        "find",
        "put",
        "use",
        "open",
        "go",
        "fetch",
        "reach",
        "unlock",
        "traverse",
    ]

    extra_words = [
        "up",
        "the",
        "a",
        "at",
        ",",
        "square",
        "and",
        "then",
        "to",
        "of",
        "rooms",
        "near",
        "opening",
        "must",
        "you",
        "matching",
        "end",
        "hallway",
        "object",
        "from",
        "room",
    ]

    all_words = colors + objects + verbs + extra_words
    assert len(all_words) == len(set(all_words))
    return {word: i for i, word in enumerate(all_words)}

# Map of agent direction, 0: East; 1: South; 2: West; 3: North
DIRECTION = {
    0: [1, 0],
    1: [0, 1],
    2: [-1, 0],
    3: [0, -1],
}

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
} 
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

# Map of skill names to integers
SKILL_TO_IDX = {"explore": 0, "go to object": 1, "pickup": 2, "drop": 3, "toggle": 4}
IDX_TO_SKILL = dict(zip(SKILL_TO_IDX.values(), SKILL_TO_IDX.keys()))




class Base_Mediator(ABC):
    """The base class for Base_Mediator."""

    def __init__(self, soft, LLM_noise = "gaussian", LLM_noise_ratio = 0.2):
        super().__init__()
        self.soft = soft
        self.obj_coordinate = {}
        self.LLM_noise_ratio = LLM_noise_ratio
        self.LLM_noise = LLM_noise
        self.see_items = []
        self.hold_items = [] 

    # obs to natural language
    def RL2LLM(self, obs, color_info=True):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]
        agent_dir = agent_map[agent_pos[0],agent_pos[1]]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)

        carrying = "nothing"
        if len(door_list):
            for door in door_list:
                i, j = door
                if color_info:
                    color = obs[i,j,1]
                    obj = f"{IDX_TO_COLOR[color]} door"
                else:
                    obj = "door"
                
                context += f"<{obj}>, "
                self.obj_coordinate[obj] = (i,j)

        if len(key_list):
            for key in key_list:
                i, j = key
                if color_info:
                    color = obs[i,j,1]
                    obj = f"{IDX_TO_COLOR[color]} key"
                else:
                    obj = "key"

                if (agent_pos == key).all():
                    carrying = obj
                else:
                    context += f"<{obj}>, " 
                    self.obj_coordinate[obj] = (i,j)

        if self.LLM_noise == "gaussian":
            probs_hold = []
            for i in range(len(self.hold_items)):
                noise = np.random.normal(0.5, 0.2)
                noise = np.clip(noise, 0, 1)
                noise = noise*self.LLM_noise_ratio
                if carrying == self.hold_items[i]:
                    noise += 1
                probs_hold.append(noise.item())
            probs_hold = [x / sum(probs_hold) for x in probs_hold]

            probs_see = []
            for i in range(len(self.see_items)):
                noise = np.random.normal(0.5, 0.2)
                noise = np.clip(noise, 0, 1)
                noise = noise*self.LLM_noise_ratio
                if carrying == self.see_items[i]:
                    noise += 1
                probs_see.append(noise.item())
            probs_see = [x / sum(probs_see) for x in probs_see]
            
            index_hold = np.random.choice(len(probs_hold), p=probs_hold)
            index_see = np.random.choice(len(probs_see), p=probs_see)

            carrying = self.hold_items[index_hold]
            context = self.see_items[index_see]
            context = f"<{context}>, "
            

        if context == '':
            context += "<nothing>, "
        context += f"holds <{carrying}>."
        
        context = f"Agent sees {context}"
        return context

    def LLM2RL(self, plans, probs):
        if self.soft:
            skill_list = [self.parser(plan) for plan in plans]
        else:
            plan = np.random.choice(plans, p=probs)
            skill_list = [self.parser(plan)]
            probs = [1.]
                
        return skill_list, probs
    
    def reset(self):
        self.obj_coordinate = {}

class SimpleDoorKey_Mediator(Base_Mediator):
    def __init__(self, soft, LLM_noise = "False", LLM_noise_ratio = 0.2):
        super().__init__(soft, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio)
        self.see_items = ["nothing", "door", "key"]
        self.hold_items = ["nothing", "key"]

    def RL2LLM(self, obs):
        return super().RL2LLM(obs, color_info=False)
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door"]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    coordinate = self.obj_coordinate["key"]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None

            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list

class KeyInBox_Mediator(Base_Mediator):
    def __init__(self, soft, LLM_noise = "False", LLM_noise_ratio = 0.2):
        super().__init__(soft, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio)
        self.see_items = ["nothing", "door", "key", "box"]
        self.hold_items = ["nothing", "key"]
       
    # ## obs2text
    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)
        box_list = np.argwhere(obs_object==7)
        carrying = "nothing"
        
        if len(key_list):
            for key in key_list:
                i, j = key
                object = "key"
                if (agent_pos == key).all():
                    carrying = object
                else:
                    context += f"<{object}>, " 
                    self.obj_coordinate[object] = (i,j)
        if len(box_list):
            for box in box_list:
                i, j = box
                object = "box"
                context += f"<{object}>, " 
                self.obj_coordinate[object] = (i,j)
                
        if len(door_list):
            for door in door_list:
                i, j = door
                object = "door"
                context += f"<{object}>, "
                self.obj_coordinate[object] = (i,j)
                
        if self.LLM_noise == "gaussian":
            probs_hold = []
            for i in range(len(self.hold_items)):
                noise = np.random.normal(0.5, 0.2)
                noise = np.clip(noise, 0, 1)
                noise = noise*self.LLM_noise_ratio
                if carrying == self.hold_items[i]:
                    noise += 1
                probs_hold.append(noise.item())
            probs_hold = [x / sum(probs_hold) for x in probs_hold]

            probs_see = []
            for i in range(len(self.see_items)):
                noise = np.random.normal(0.5, 0.2)
                noise = np.clip(noise, 0, 1)
                noise = noise*self.LLM_noise_ratio
                if carrying == self.see_items[i]:
                    noise += 1
                probs_see.append(noise.item())
            probs_see = [x / sum(probs_see) for x in probs_see]
            
            index_hold = np.random.choice(len(probs_hold), p=probs_hold)
            index_see = np.random.choice(len(probs_see), p=probs_see)

            carrying = self.hold_items[index_hold]
            context = self.see_items[index_see]
            context = f"<{context}>, "
                
        if context == '':
            context += "<nothing>, "
            
        context += f"holds <{carrying}>."
        context = f"Agent sees {context}"
        
        return context
    
    def parser(self, plan):
        # action:
        skill_list = []
        plan = plan.replace("and", ",")
        skills = plan.split(',')
        #print(f"skills = {skills}")
        for i in range(len(skills)) :
            skills[i] = skills[i].lower()
        for text in skills :
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "toggle" in text or "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                #print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    if "key" in self.obj_coordinate.keys():
                        coordinate = self.obj_coordinate["key"]
                    else:
                        coordinate = self.obj_coordinate["box"]
                elif "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door"]
                elif "box" in text:
                    obj = OBJECT_TO_IDX["box"]
                    coordinate = self.obj_coordinate["box"]
                else:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
            except Exception as e:
                print("Unknown Planning :", text)
                print("error = ", e)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
            
            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)       
        # print(skill_list) 
    
        return skill_list

class RandomBoxKey_Mediator(Base_Mediator):
    def __init__(self, soft, LLM_noise = "False", LLM_noise_ratio = 0.2):
        super().__init__(soft, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio)
        self.see_items = ["nothing", "door", "key", "box"]
        self.hold_items = ["nothing", "key"]
       
    # ## obs2text
    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)
        box_list = np.argwhere(obs_object==7)
        carrying = "nothing"
        
        if len(key_list):
            for key in key_list:
                i, j = key
                object = "key"
                if (agent_pos == key).all():
                    carrying = object
                else:
                    context += f"<{object}>, " 
                    self.obj_coordinate[object] = (i,j)
        if len(box_list):
            for box in box_list:
                i, j = box
                object = "box"
                context += f"<{object}>, " 
                self.obj_coordinate[object] = (i,j)
                
        if len(door_list):
            for door in door_list:
                i, j = door
                object = "door"
                context += f"<{object}>, "
                self.obj_coordinate[object] = (i,j)
                
        if self.LLM_noise == "gaussian":
            probs_hold = []
            for i in range(len(self.hold_items)):
                noise = np.random.normal(0.5, 0.2)
                noise = np.clip(noise, 0, 1)
                noise = noise*self.LLM_noise_ratio
                if carrying == self.hold_items[i]:
                    noise += 1
                probs_hold.append(noise.item())
            probs_hold = [x / sum(probs_hold) for x in probs_hold]

            probs_see = []
            for i in range(len(self.see_items)):
                noise = np.random.normal(0.5, 0.2)
                noise = np.clip(noise, 0, 1)
                noise = noise*self.LLM_noise_ratio
                if carrying == self.see_items[i]:
                    noise += 1
                probs_see.append(noise.item())
            probs_see = [x / sum(probs_see) for x in probs_see]
            
            index_hold = np.random.choice(len(probs_hold), p=probs_hold)
            index_see = np.random.choice(len(probs_see), p=probs_see)

            carrying = self.hold_items[index_hold]
            context = self.see_items[index_see]
            context = f"<{context}>, "
                
        if context == '':
            context += "<nothing>, "
            
        context += f"holds <{carrying}>."
        context = f"Agent sees {context}"
        
        return context
    
    def parser(self, plan):
        # action:
        skill_list = []
        plan = plan.replace("and", ",")
        skills = plan.split(',')
        #print(f"skills = {skills}")
        for i in range(len(skills)) :
            skills[i] = skills[i].lower()
        for text in skills :
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "toggle" in text or "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                #print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    if "key" in self.obj_coordinate.keys():
                        coordinate = self.obj_coordinate["key"]
                    else:
                        coordinate = self.obj_coordinate["box"]
                elif "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door"]
                elif "box" in text:
                    obj = OBJECT_TO_IDX["box"]
                    coordinate = self.obj_coordinate["box"]
                else:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
            except Exception as e:
                print("Unknown Planning :", text)
                print("error = ", e)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
            
            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)       
        # print(skill_list) 
    
        return skill_list
        
class ColoredDoorKey_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        return super().RL2LLM(obs)
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "door" in text:
                    obj = OBJECT_TO_IDX["door"]
                    words = text.split(' ')
                    filter_words = []
                    for w in words:
                        w1="".join(c for c in w if c.isalpha())
                        filter_words.append(w1)
                    object_word = filter_words[-2] + " " + filter_words[-1]
                    coordinate = self.obj_coordinate[object_word]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]    
                    words = text.split(' ')
                    filter_words = []
                    for w in words:
                        w1="".join(c for c in w if c.isalpha())
                        filter_words.append(w1)
                    object_word = filter_words[-2] + " " + filter_words[-1]
                    coordinate = self.obj_coordinate[object_word]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                print("Unknown Planning :", text)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None
                
            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list
    
class TwoDoor_Mediator(Base_Mediator):
    def __init__(self, soft):
        super().__init__(soft)

    def RL2LLM(self, obs):
        context = ''
        if len(obs.shape) == 4:
            obs = obs[0,:,:,-4:]
        obs_object = copy.deepcopy(obs[:,:,0])
        agent_map = obs[:, :, 3]
        agent_pos = np.argwhere(agent_map != 4)[0]
        agent_dir = agent_map[agent_pos[0],agent_pos[1]]

        key_list = np.argwhere(obs_object==5)
        door_list = np.argwhere(obs_object==4)

        carrying = "nothing"
        if len(key_list):
            for key in key_list:
                i, j = key
                obj = "key"

                if (agent_pos == key).all():
                    carrying = obj
                else:
                    context += f"<{obj}>, " 
                    self.obj_coordinate[obj] = (i,j)

        if len(door_list):
            n = 1
            for door in door_list:
                i, j = door
                obj = f"door{n}"
                n += 1
                
                context += f"<{obj}>, "
                self.obj_coordinate[obj] = (i,j)

        if context == '':
            context += "<nothing>, "
        context += f"holds <{carrying}>."
        
        context = f"Agent sees {context}"
        return context
    
    def parser(self, plan):
        skill_list = []
        skills = plan.split(',')
        for text in skills:
            # action:
            if "explore" in text:
                act = SKILL_TO_IDX["explore"]
            elif "go to" in text:
                act = SKILL_TO_IDX["go to object"]
            elif "pick up" in text:
                act = SKILL_TO_IDX["pickup"]
            elif "drop" in text:
                act = SKILL_TO_IDX["drop"]
            elif "open" in text:
                act = SKILL_TO_IDX["toggle"]
            else:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
            # object:
            try:
                if "door1" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door1"]
                elif "door2" in text:
                    obj = OBJECT_TO_IDX["door"]
                    coordinate = self.obj_coordinate["door2"]
                elif "key" in text:
                    obj = OBJECT_TO_IDX["key"]
                    coordinate = self.obj_coordinate["key"]
                elif "explore" in text:
                    obj = OBJECT_TO_IDX["empty"]
                    coordinate = None
                else:
                    assert False
            except:
                # print("Unknown Planning :", text)
                act = 6 # do nothing
                obj = OBJECT_TO_IDX["empty"]
                coordinate = None

            skill = {"action": act,
                     "object": obj,
                     "coordinate": coordinate,}
            skill_list.append(skill)
        
        return skill_list


if __name__ == "__main__":
    word = get_minigrid_words()
