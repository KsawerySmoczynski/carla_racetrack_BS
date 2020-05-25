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
import torch
from torch import nn
from control.abstract_control import Controller

#Local imports
from net.a3c_lstm_net import A3C_LSTM


class NNController(Controller):
    def __init__(self, net:nn.Module, device:str='cuda:0'):
        super(NNController, self).__init__()
        self.net = net
        self.device = torch.device(device)

    def preprocess(self, state:dict, lstm_data:bool=True):
        '''
        Method converting data from state form to net form.
        :param state: dict
        :param lstm_data: bool, whether to take lstm hx and cx from state or generate own.
        :return:
        '''
        if lstm_data:
            hx, cx = state['hx'], state['cx']
        else:
            hx, cx = (torch.zeros(size=(1, self.net.lstm.hidden_size), device=self.device),
                      torch.zeros(size=(1, self.net.lstm.hidden_size), device=self.device))

        if 'rgb' in state:
            rgb = torch.from_numpy(state['rgb_data']).to(device=self.device)
        else:
            rgb = None

        #Add preprocess -> reshaping, normalizing, dimensions reduction
        # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python -> 1 layer tensor
        depth = torch.from_numpy(state['depth_data']).to(device=self.device)

        numeric = torch.tensor([state['gas_brake'], state['steer'], state['distance_2finish'], state['velocity']],
                               device=self.device)


        return numeric, depth, rgb, (hx, cx)

    def control(self, state, **kwargs):

        pass

# class NnA2CController(nn.Module, Controller):
#     def __init__(self, frames_shape):
#         """Setus up all of the neural networks necessary for the controller to work
#
#         Parameters
#         ----------
#         frames_shape
#             the shape of the multidimensional carla data fetched from an agent's sensor
#
#         Attributes
#         ----------
#         conv_net: nn.Sequential
#             a neural network responsible for extracting the current state of the environment, whose output is meant to serve as input for policy and critic networks
#         actor_net: nn.Sequential
#             a neural network responsible for the current agent's policy
#         critic_net: nn.Sequential
#             a neural network responsible for approximating the advantage function regarding the action space
#
#         Methods
#         -------
#         forward
#             method feeding the data through the policy and critic nets after preprocessing sensor data using the CNN
#         """
#         super(NnA2CController, self).__init__()
#
#         #lenet inspired net upscaled due to carla frames being bigger than minst digits ;)
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(in_channels=frames_shape[0], out_channels=128, kernel_size=7, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=4),
#             nn.Conv2d(in_channels=64, out_channels=120, kernel_size=3, stride=1),
#             nn.Tanh()
#         )
#
#         self.conv_out_size = int(np.prod(self.conv_net(torch.zeros(1, *frames_shape)).size()))
#
#         self.actor_net = nn.Sequential(
#             nn.Linear(self.conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2) #2 returned values define the action taken under the current policy, which consists of the gas/break pedal and the steering angle
#         )
#
#         self.critic_net = nn.Sequential(
#             nn.Linear(self.conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1) #a single return - the state value
#         )
#
#
#     def forward(self, sensor_data):
#         conv_output = self.conv_net(sensor_data).view(sensor_data.size()[0], -1)
#         return self.actor_net(conv_output), self.critic_net(conv_output)
#
#     def control(self, state, pts_3D):
#
#         #leaving the following state elements here, but intend to use just camera data for starters
#         location = state['location']
#         x, y = location[0], location[1]
#         v = state['velocity'] # km / h #how to get speed?????????
#         ψ = np.radians(state['yaw']) #adding 180 as carla returns yaw degrees in (-180, 180) range
#
#         actor_out, critic_out = self.forward(torch.cat((state['depth'], state['rgb']), 1))
#
#         actions = {
#             'steer': int(actor_out[0][0]),
#             'gas_brake':  int(actor_out[0][1]),
#             'state_value': int(critic_out[0][0])
#         }
#
#         return actions

class A3C_LSTM_Controller(NNController):
    def __init__(self, net:A3C_LSTM, device:str= 'cuda:0'):
        super(A3C_LSTM_Controller, self).__init__(net=net, device=device)

    def control(self, state, **kwargs):
        x_numeric, depth, rgb, (hx, cx) = self.preprocess(state=state)
        x = x_numeric, depth, rgb, (hx, cx)
        reward, actions, (hx, cx) = self.net.forward(x)
        return {'gas_brake':actions[0], 'steer':actions[1], 'hx':hx, 'cx':cx, 'reward':reward}


class A3C_LSTM_Imitation(NNController):
    def __init__(self, net:A3C_LSTM, device:str= 'cuda'):
        super(A3C_LSTM_Imitation, self).__init__(net=net, device=device)

    def control(self, state, **kwargs):
        x_numeric, depth, rgb, (hx, cx) = self.preprocess(state=state, lstm_data=False)

        x = (x_numeric, depth, rgb, (hx, cx)) #converted state to net form
        reward, actions, (hx, cx) = self.net.forward(x)
        return {'gas_brake':actions[0], 'steer':actions[1], 'hx':hx, 'cx':cx, 'predicted_reward':reward}


