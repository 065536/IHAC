import numpy as np
from planner import Planner
from skill import GoTo_Goal, Explore, Pickup, Drop, Toggle, Wait
from mediator import IDX_TO_SKILL, IDX_TO_OBJECT

# single step (can handle soft planner)
class MetaController():
    def __init__(self, task, offline, soft, prefix, action_space, agent_view_size, LLM_noise = "False", LLM_noise_ratio = 0.2, pi_meta_controller_noise = "gaussian", noise_ratio = 0.2):
        self.planner = Planner(task, offline, soft, prefix, LLM_noise = LLM_noise, LLM_noise_ratio = LLM_noise_ratio, pi_meta_controller_noise = pi_meta_controller_noise, noise_ratio = noise_ratio)
        self.agent_view_size = agent_view_size
        self.action_space = action_space
        
    def get_skill_name(self, skill):
        try:
            return IDX_TO_SKILL[skill["action"]] + " " + IDX_TO_OBJECT[skill["object"]]
        except AttributeError:
            return "None"
        
    def reset(self):
        self.skill = None
        self.skill_list = []
        self.skill_teminated = False
        self.planner.reset() 

    def skill2meta_controller(self, skill):
        '''
        0: explore
        1: go to goal
        2: pick up object
        3: drop object
        4: toggle object
        6: wait
        '''
        skill_action = skill['action']
        if skill_action == 0:
            meta_controller = Explore(self.agent_view_size)
        elif skill_action == 1:
            meta_controller = GoTo_Goal(skill['coordinate'])
        elif skill_action == 2:
            meta_controller = Pickup(skill['object'])
        elif skill_action == 3:
            meta_controller = Drop(skill['object'])
        elif skill_action == 4:
            meta_controller = Toggle(skill['object'])
        elif skill_action == 6:
            meta_controller = Wait()
        else:
            assert False, "invalid skill"
        return meta_controller
    
    def get_action(self, skill_list, obs):
        teminated = True
        action = None
        while not action and teminated and len(skill_list) > 0:
            skill = skill_list.pop(0)
            meta_controller = self.skill2meta_controller(skill)
            action, teminated = meta_controller(obs)
                
        if action == None:

            action = 6
            
        action = np.array([i == action for i in range(self.action_space)], dtype=np.float32)
            
        return action
    
    def __call__(self, obs):
        skill_list, probs = self.planner(obs)
        action = np.zeros(self.action_space)
        for skills, prob in zip(skill_list, probs):
            action += self.get_action(skills, obs) * prob
        return action


