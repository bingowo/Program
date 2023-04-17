import gym
from gym import spaces
from KnowledgeTracing.model.SAINTModel import saint
from KnowledgeTracing.Constant import Constants as C
import numpy as np
import torch

class Car2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self):
        self.action_space = spaces.Discrete(136) # 0, 1, 2，3，4: 不动，上下左右
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,136))

        self.ques        = torch.zeros([1, C.MAX_STEP]).cuda()
        self.concepts    = torch.zeros([1, C.MAX_STEP, 4]).cuda()
        self.ans         = torch.zeros([1, C.MAX_STEP]).cuda()
        self.features         = torch.zeros([1, C.MAX_STEP]).cuda()

        self.ques = self.ques.long()
        self.concepts = self.concepts.long()
        self.ans = self.ans.long()
        self.lens = 1
        self.ques[0,0] = torch.randint(1,136,(1,1))
        self.ans[0,0] = torch.randint(0,1,(1,1))

        self.state = None

        self.model = saint(dim_model=128,
                        num_en=2,
                        num_de=2,
                        heads_en=8,
                        heads_de=8,
                        total_ex=C.exer_n,
                        total_cat=C.knowledge_n+1,
                        total_in=2,
                        seq_len=C.MAX_STEP,
                        dropout=0.3
                        ).cuda()

        state_dict=torch.load('net_params.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def cal(self):
        pred = self.model(self.ques,self.concepts,self.ans,self.features).cpu()
        return pred.detach().numpy()[:136]
    
    def ps(self):
        tmp = np.zeros([136])
        for i in range(136):
            action = i+1
            self.ques[0,self.lens] = action

            result = self.cal()[0,self.lens,0]

            tmp[i] = float(result)
        return tmp

        
    
    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        _r = np.mean(self.state)

        action = int(action)
        action += 1

        self.ques[0,self.lens] = action
        result = self.cal()[0,self.lens,0]

        # print(result)
        if result > np.random.uniform():
            result = 1
        else:
            result = 0

        self.ans[0,self.lens] = result

        self.lens += 1

        self.state = self.ps()
        self.counts += 1
            
        done = (self.lens >= 49)
        done = bool(done)
        
        reward = np.mean(self.state) - _r
            
        return self.state, reward, done, {}
    
    def reset(self):
        self.ques        = torch.zeros([1, C.MAX_STEP]).cuda()
        self.ans         = torch.zeros([1, C.MAX_STEP]).cuda()

        self.ques = self.ques.long()
        self.ans = self.ans.long()
        self.lens = 5
        self.ques[0,0:self.lens] = torch.randint(1,136,(1,self.lens))
        # self.ans[0,0:self.lens] = torch.randint(0,1,(1,self.lens))
        self.ans[0,0:self.lens] = torch.ones([1,self.lens])


        self.state = self.ps()
        self.counts = 0
        return self.state
        
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
        
if __name__ == '__main__':
    env = Car2DEnv()
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)