import numpy as np
import torch
from torch import nn
from PIL import ImageFilter
from torchvision.transforms import transforms

from config import NUMERIC_FEATURES
from control.abstract_control import Controller
from net.ddpg_net import DDPGActor, DDPGCritic
from net.utils import img_to_pil


class NNController(Controller):
    def __init__(self, actor_net:DDPGActor, critic_net:DDPGCritic, features:list=NUMERIC_FEATURES, device:str='cuda:0'):
        super(NNController, self).__init__()
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.device = torch.device(device)
        self.features = features
        self.transform = transforms.ToTensor()

    def dict(self):
        controller = {'actor_net': self.actor_net.name,
                      'critic_net': self.critic_net.name,
                      'device': str(self.device),
                      'features': self.features,
                      'transform': repr(self.transform)}
        return controller

    def preprocess(self, state:dict):
        '''

        :param state:
        :return:
        '''
        x_numeric = torch.Tensor([state[feature] for feature in self.features]).unsqueeze(0).float().to(self.device)
        imgs = [self.transform(img_to_pil(img)) for img in state['depth_data']]
        depth = torch.cat(imgs, dim=2).unsqueeze(0).float().to(self.device)
        # img = img - img.mean()

        return (x_numeric, depth)

    def control(self, state, **kwargs):
        input = self.preprocess(state)
        action = self.actor_net(input).view(-1)
        # q_pred = self.critic_net((action, *input)).view(-1)

        action = {
            'steer': round(float(action[0]), 3),
            'gas_brake': round(float(action[1]), 3),
            # 'q_pred': q_pred
        }

        return action




# Torch multiprocessing
# https://pytorch.org/docs/stable/notes/multiprocessing.html

# Q-learning
# https://arxiv.org/pdf/1903.10605.pdf

# A2C
# https://arxiv.org/abs/1903.11329 <- old
# https://arxiv.org/abs/1903.11329 <- nówerka

# A3C
# https://esc.fnwi.uva.nl/thesis/centraal/files/f285129090.pdf
# https://github.com/dgriff777/a3c_continuous/blob/master/model.py
# https://github.com/devendrachaplot/DeepRL-Grounding

# DDPG
# https://arxiv.org/pdf/1804.08617.pdf
# https://ai.stackexchange.com/questions/6317/what-is-the-difference-between-on-and-off-policy-deterministic-actor-critic ważne do tego wyżej
# http://proceedings.mlr.press/v32/silver14.pdf
# https://cardwing.github.io/files/RL_course_report.pdf
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html -> IMPLEMENTACJA !!!!!!!!!!!!!!!!!!1
# https://www.ijcai.org/Proceedings/2018/0444.pdf -> dojebańsza wersja
# https://arxiv.org/pdf/1903.00827.pdf -> MEGA DOJEBAŃSZA WERSJA!!!!!!!!!!!!

# Off-policy
# https://towardsdatascience.com/the-false-promise-of-off-policy-reinforcement-learning-algorithms-c56db1b4c79a

# ZABIERZ SOBIE BUILDING BLOCKI Z OPEN AI
# https://github.com/openai/spinningup/tree/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg -> DDPG
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ddpg/core.py#L35
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

# Roadmap:
# 1 Skopiuj ddpg z openAI i książki, https://www.udemy.com/course/cutting-edge-artificial-intelligence/learn/lecture/14460554#overview
# 2. Przeczytaj papiery z dojebańszą wersją
# 3 Zrób osobny model (duplikuj i ulepsz)


# NAGRODA
# Bierzemy co 0.5% punkt do liczenia nagrody jako odległości od końca toru
# Array pokazujący okrążenie, array w którym dodajemy minięte punkty na jego koniec i array do liczenia nagród

#3-rd party imports
