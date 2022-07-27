import json
import os
import sys
import numpy as np
import random
import math
import time
import wandb
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.logger import print_progress
from tqdm import tqdm

class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path'], 'pred_objid': v['pred_objid']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

    def test_viz(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout_viz(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    env_actions = {
      'left': (0, -1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0, -1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    #def __init__(self, args, env, rank=0, adv_training=True,\
    #        adv_delta_coarse=1, adv_delta_txt=2, adv_delta_fine=3):
    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) 

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        raise NotImplementedError('child class should implement _build_model: self.vln_bert & self.critic')

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, viz=False):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        if viz:
            super().test_viz(iters=iters)
        else:
            super().test(iters=iters)

    #def train(self, n_iters, feedback='teacher', adv_training=True,\
    #        adv_delta_coarse=None, adv_delta_txt=None, adv_delta_fine=None,\
    #        **kwargs):# Train interval iters
    def train(self, n_iters, feedback='teacher', use_mat=False, adv_step=4,\
            adv_loss_weight=1.5, **kwargs):# Train interval iters
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []

        for iter in tqdm(range(1, n_iters + 1)):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0
            
            adv_training = False
            coarse_delta = torch.zeros(1, 1, 768, requires_grad = True, device="cuda")
            txt_delta = torch.zeros(1, 1, 768, requires_grad = True, device="cuda")
            fine_delta = torch.zeros(1, 1, 768, requires_grad = True, device="cuda")

            if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                _, nav, obj = self.rollout(train_ml=1., train_rl=False, reset=False, adv_training=adv_training,\
                            adv_delta_coarse=None, adv_delta_txt=None,\
                            adv_delta_fine=None, **kwargs)
            elif self.args.train_alg == 'dagger': 
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    _, nav, obj = self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, adv_training=adv_training,\
                        adv_delta_coarse=None, adv_delta_txt=None,\
                        adv_delta_fine=None, **kwargs)
                self.feedback = self.args.dagger_sample
                _, nav, obj = self.rollout(train_ml=1., train_rl=False, reset=False, adv_training=adv_training,\
                            adv_delta_coarse=None, adv_delta_txt=None,\
                            adv_delta_fine=None, **kwargs)
            else:
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    _, nav, obj = self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, adv_training=adv_training,\
                        adv_delta_coarse=None, adv_delta_txt=None,\
                        adv_delta_fine=None, **kwargs)
                self.feedback = 'sample'
                _, nav, obj = self.rollout(train_ml=None, train_rl=True, reset=False, adv_training=adv_training,\
                            adv_delta_coarse=None, adv_delta_txt=None,\
                            adv_delta_fine=None, **kwargs)

            #print(f"nav:{nav.size()}")
            #print(f"obj:{obj.size()}")

            
            if use_mat:
                self.loss = 0
                adv_training = True
                coarse_v, txt_v, fine_v = 0, 0, 0
                coarse_s, txt_s, fine_s = 0, 0, 0

                for astep in range(adv_step):
                    coarse_delta.requires_grad_()
                    txt_delta.requires_grad_()
                    fine_delta.requires_grad_()

                    if self.args.train_alg == 'imitation':
                        self.feedback = 'teacher'
                        _, adv_nav, adv_obj = self.rollout(train_ml=1., train_rl=False, reset=False, adv_training=adv_training,\
                                    adv_delta_coarse=coarse_delta, adv_delta_txt=txt_delta,\
                                    adv_delta_fine=fine_delta, **kwargs)
                    elif self.args.train_alg == 'dagger': 
                        if self.args.ml_weight != 0:
                            self.feedback = 'teacher'
                            _, adv_nav, adv_obj = self.rollout(
                                train_ml=self.args.ml_weight, train_rl=False, adv_training=adv_training,\
                                adv_delta_coarse=coarse_delta, adv_delta_txt=txt_delta,\
                                adv_delta_fine=fine_delta, **kwargs)
                        self.feedback = self.args.dagger_sample
                        _, adv_nav, adv_obj = self.rollout(train_ml=1., train_rl=False, reset=False, adv_training=adv_training,\
                                    adv_delta_coarse=coarse_delta, adv_delta_txt=txt_delta,\
                                    adv_delta_fine=fine_delta, **kwargs)
                    else:
                        if self.args.ml_weight != 0:
                            self.feedback = 'teacher'
                            _, adv_nav, adv_obj = self.rollout(
                                train_ml=self.args.ml_weight, train_rl=False, adv_training=adv_training,\
                                adv_delta_coarse=coarse_delta, adv_delta_txt=txt_delta,\
                                adv_delta_fine=fine_delta, **kwargs)
                        self.feedback = 'sample'
                        _, adv_nav, adv_obj = self.rollout(train_ml=None, train_rl=True, reset=False, adv_training=adv_training,\
                                    adv_delta_coarse=coarse_delta, adv_delta_txt=txt_delta,\
                                    adv_delta_fine=fine_delta, **kwargs)


                    #print(f"nav2:{adv_nav.size()}")
                    #print(f"obj2:{adv_obj.size()}")
                    #nav_kl_loss = F.kl_div(nav, adv_nav.clone().detach(), reduction='none') + \
                    #    F.kl_div(nav.clone().detach(), adv_nav, reduction='none')

                    #obj_kl_loss = F.kl_div(obj, adv_obj.clone().detach(), reduction='none') + \
                    #    F.kl_div(obj.clone().detach(), adv_obj, reduction='none')

                    #total_loss = (self.loss + adv_loss_weight * (nav_kl_loss.mean())) / adv_step
                    #total_loss = (self.loss + adv_loss_weight * (nav_kl_loss.mean() +\
                    #            obj_kl_loss.mean())) / adv_step
                    #total_loss.backward(retain_graph=True)
                    #self.loss += adv_loss_weight * (nav_kl_loss.mean())
                    self.loss.backward(retain_graph=True)
                    #self.loss.backward()

                    if astep == adv_step - 1:
                        break

                    # coarse
                    #print(coarse_delta.requires_grad_())
                    #print(coarse_delta.grad)
                    coarse_delta_grad = coarse_delta.grad.clone().detach().float()
                    #print("###############")
                    #print(coarse_delta_grad)
                    denorm = torch.norm(coarse_delta_grad.view(coarse_delta_grad.size(0), -1), dim=1).view(-1, 1)
                    #print(denorm)
                    denorm = torch.clamp(denorm, min=1e-8)
                    #print(denorm)
                    denorm = torch.unsqueeze(denorm, 2)
                    #print(denorm)
                    coarse_g = coarse_delta_grad / denorm
                    beta1, beta2 = 0.9, 0.9
                    coarse_v = beta1 * coarse_v + (1 - beta1) * coarse_g
                    coarse_s = beta2 * coarse_s + (1 - beta2) * coarse_g ** 2
                    coarse_v = coarse_v / (1 - beta1 ** (astep + 1))
                    coarse_s = coarse_s / (1 - beta2 ** (astep + 1))
                    denorm = torch.norm(coarse_s.view(coarse_s.size(0), -1), dim=1).view(-1, 1)
                    #print(denorm)
                    denorm = torch.clamp(denorm, min=1e-8)
                    #print(denorm)
                    denorm = torch.unsqueeze(denorm, 2)
                    #print(denorm)
                    coarse_delta_step = (1e-3 * coarse_v / denorm).to(coarse_delta)
                    coarse_delta = (coarse_delta + coarse_delta_step).detach()


                    ## txt
                    #txt_delta_grad = txt_delta.grad.clone().detach().float()
                    #denorm = torch.norm(txt_delta_grad.view(txt_delta_grad.size(0), -1), dim=1).view(-1, 1)
                    #denorm = torch.clamp(denorm, min=1e-8)
                    #txt_g = txt_delta_grad / denorm
                    #beta1, beta2 = 0.9, 0.9
                    #txt_v = beta1 * txt_v + (1 - beta1) * txt_g
                    #txt_s = beta2 * txt_s + (1 - beta2) * txt_g ** 2
                    #txt_v = txt_v / (1 - beta1 ** (astep + 1))
                    #txt_s = txt_s / (1 - beta2 ** (astep + 1))
                    #denorm = torch.norm(txt_s.view(txt_s.size(0), -1), dim=1).view(-1, 1)
                    #denorm = torch.clamp(denorm, min=1e-8)
                    #txt_delta_step = (1e-3 * txt_v / denorm).to(txt_delta)
                    #txt_delta = (txt_delta + txt_delta_step).detach()
            
                    ## fine
                    #fine_delta_grad = fine_delta.grad.clone().detach().float()
                    #denorm = torch.norm(fine_delta_grad.view(fine_delta_grad.size(0), -1), dim=1).view(-1, 1)
                    #denorm = torch.clamp(denorm, min=1e-8)
                    #fine_g = fine_delta_grad / denorm
                    #beta1, beta2 = 0.9, 0.9
                    #fine_v = beta1 * fine_v + (1 - beta1) * fine_g
                    #fine_s = beta2 * fine_s + (1 - beta2) * fine_g ** 2
                    #fine_v = fine_v / (1 - beta1 ** (astep + 1))
                    #fine_s = fine_s / (1 - beta2 ** (astep + 1))
                    #denorm = torch.norm(fine_s.view(fine_s.size(0), -1), dim=1).view(-1, 1)
                    #denorm = torch.clamp(denorm, min=1e-8)
                    #fine_delta_step = (1e-3 * fine_v / denorm).to(fine_delta)
                    #fine_delta = (fine_delta + fine_delta_step).detach()

            else:
                #print(self.rank, iter, self.loss)
                self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            # if self.args.aug is None:
            #     print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                if list(model_keys)[0].startswith('module.') and (not list(load_keys)[0].startswith('module.')):
                    state_dict = {'module.'+k: v for k, v in state_dict.items()}
                same_state_dict = {}
                extra_keys = []
                for k, v in state_dict.items():
                    if k in model_keys:
                        same_state_dict[k] = v
                    else:
                        extra_keys.append(k)
                state_dict = same_state_dict
                print('Extra keys in state_dict: %s' % (', '.join(extra_keys)))
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1


