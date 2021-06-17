from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MLPBase
import torchfold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def relative_pos_encoding(children_pos, cur_pos):
    relative_pos = children_pos - cur_pos
    relative_dis = torch.abs(relative_pos)
    relative_feature = torch.cat([relative_dis, relative_pos, cur_pos, children_pos], dim=-1)
    return relative_feature


class CriticUp(nn.Module):
    """a bottom-up module used in bothway message passing that only passes message to its parent"""
    def __init__(self, state_dim, action_dim, msg_dim):
        super(CriticUp, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 64) # tgt node input message
        self.fc2 = nn.Linear(64 + msg_dim, 64) # (tgt, src nodes) message
        self.fc3 = nn.Linear(64, msg_dim) # output
        self.att_linear = nn.Linear(9 + state_dim, msg_dim)

    def forward(self, x, u, children_states, *m):
        xu = torch.cat([x, u], dim=-1)
        xu = self.fc1(xu)
        xu = F.normalize(xu, dim=-1)

        att = F.softmax(self.att_linear(children_states), dim=-2)
        m = torch.cat(m, dim=-1)
        m = m.view((*children_states.shape[:-1], -1))
        m = (m * att).sum(dim=-2)

        xum = torch.cat([xu, m], dim=-1)
        xum = torch.tanh(xum)
        xum = self.fc2(xum)
        xum = torch.tanh(xum)
        xum = self.fc3(xum)
        xum = F.normalize(xum, dim=-1)
        msg_up = xum

        return msg_up


class CriticDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs q-values"""
    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, self_input_dim, action_dim, msg_dim, max_children):
        super(CriticDownAction, self).__init__()
        self.baseQ1 = MLPBase(12 + self_input_dim + action_dim + msg_dim, 1)
        self.baseQ2 = MLPBase(12 + self_input_dim + action_dim + msg_dim, 1)
        self.msg_base = MLPBase(12 + self_input_dim + msg_dim, msg_dim * max_children)

    def forward(self, rel_pos, x, u, m):
        xum = torch.cat([rel_pos, x, u, m], dim=-1)
        x1 = self.baseQ1(xum)
        x2 = self.baseQ2(xum)
        xm = torch.cat([rel_pos, x, m], dim=-1)
        xm = torch.tanh(xm)
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)

        return x1, x2, msg_down

    def Q1(self, rel_pos, x, u, m):
        xum = torch.cat([rel_pos, x, u, m], dim=-1)
        x1 = self.baseQ1(xum)
        xm = torch.cat([rel_pos, x, m], dim=-1)
        xm = torch.tanh(xm)
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)

        return x1, msg_down


class CriticGraphPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""
    def __init__(self, state_dim, action_dim, msg_dim, batch_size, max_children, disable_fold):
        super(CriticGraphPolicy, self).__init__()
        self.num_limbs = 1
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        assert self.action_dim == 1
        
        # bottom-up then top-down
        self.sNet = nn.ModuleList([CriticUp(state_dim, action_dim, msg_dim)] * self.num_limbs).to(device)
        if not self.disable_fold:
            for i in range(self.num_limbs):
                setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])

        # we pass msg_dim as first argument because in both-way message-passing, each node takes in its passed-up message as 'state'
        self.critic = nn.ModuleList([CriticDownAction(msg_dim, action_dim, msg_dim, max_children)] * self.num_limbs).to(device)
        if not self.disable_fold:
            for i in range(self.num_limbs):
                setattr(self, "critic" + str(i).zfill(3), self.critic[i])

        if not self.disable_fold:
            for i in range(self.max_children):
                setattr(self, 'get_{}'.format(i), self.addFunction(i))

    def forward(self, state, action):
        self.clear_buffer()
        if not self.disable_fold:
            self.fold = torchfold.Fold()
            self.fold.cuda()
            self.zeroFold_td = self.fold.add("zero_func_td")
            self.zeroFold_bu = self.fold.add("zero_func_bu")
            self.x1_fold, self.x2_fold = [], []
        assert state.shape[1] == self.state_dim * self.num_limbs, 'state.shape[1] expects {} but got {} with num_limbs being {} and state_dim being {}'.format(self.state_dim * self.num_limbs, state.shape[1], self.num_limbs, self.state_dim)
        for i in range(self.num_limbs):
            self.input_state[i] = state[:, i * self.state_dim:(i + 1) * self.state_dim]
            self.input_action[i] = action[:, i]
            self.input_action[i] = torch.unsqueeze(self.input_action[i], -1)
            if not self.disable_fold:
                self.input_state[i] = torch.unsqueeze(self.input_state[i], 0)
                self.input_action[i] = torch.unsqueeze(self.input_action[i], 0)

        # bottom up transmission by recursion
        for i in range(self.num_limbs):
            self.bottom_up_transmission(i)

        # top down transmission by recursion
        for i in range(self.num_limbs):
            self.top_down_transmission(i)

        if not self.disable_fold:    
            self.x1_fold = self.x1_fold + self.x1
            self.x2_fold = self.x2_fold + self.x2
            self.x1, self.x2 = self.fold.apply(self, [self.x1_fold, self.x2_fold])
            self.x1 = torch.transpose(self.x1, 0, 1)
            self.x2 = torch.transpose(self.x2, 0, 1)
            self.fold = None
        else:
            self.x1 = torch.stack(self.x1, dim=-1)  # (bs,num_limbs,1)
            self.x2 = torch.stack(self.x2, dim=-1)

        return torch.sum(self.x1, dim=-1), torch.sum(self.x2, dim=-1)

    def Q1(self, state, action):
        self.clear_buffer()
        if not self.disable_fold:
            self.fold = torchfold.Fold()
            self.fold.cuda()
            self.zeroFold_td = self.fold.add("zero_func_td")
            self.zeroFold_bu = self.fold.add("zero_func_bu")
            self.x1_fold = []

        for i in range(self.num_limbs):
            self.input_state[i] = state[:, i * self.state_dim:(i + 1) * self.state_dim]
            self.input_action[i] = action[:, i]
            self.input_action[i] = torch.unsqueeze(self.input_action[i], -1)
            if not self.disable_fold:
                self.input_state[i] = torch.unsqueeze(self.input_state[i], 0)
                self.input_action[i] = torch.unsqueeze(self.input_action[i], 0)

        # bottom up transmission by recursion
        for i in range(self.num_limbs):
            self.bottom_up_transmission(i)

        # top down transmission by recursion
        for i in range(self.num_limbs):
            self.top_down_transmission(i)

        if not self.disable_fold:
            self.x1_fold = self.x1_fold + self.x1
            self.x1 = self.fold.apply(self, [self.x1_fold])[0]
            self.x1 = torch.transpose(self.x1, 0, 1)
            self.fold = None
        else:
            self.x1 = torch.stack(self.x1, dim=-1)  # (bs,num_limbs,1)

        return torch.sum(self.x1, dim=-1)

    def bottom_up_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_bu
            else:
                return torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device)

        if self.msg_up[node] is not None:
            return self.msg_up[node]

        state = self.input_state[node]
        action = self.input_action[node]

        cur_pos, _ = state.split([3, state.shape[-1] - 3], dim=-1)

        children = [i for i, x in enumerate(self.parents) if x == node]
        assert (self.max_children - len(children)) >= 0
        children += [-1] * (self.max_children - len(children))
        msg_in = [None] * self.max_children
        children_states = [None] * self.max_children

        for i in range(self.max_children):
            # children[i] = children node idx
            if children[i] == -1:
                children_states[i] = torch.zeros((*cur_pos.shape[:-1], 9 + state.shape[-1])).to(device)
            else:
                child_state = self.input_state[children[i]]
                children_pos, children_f  = child_state.split([3, child_state.shape[-1] - 3], dim=-1)

                rel_pos = relative_pos_encoding(children_pos, cur_pos)

                children_states[i] = torch.cat((rel_pos, children_f), dim=-1)

            msg_in[i] = self.bottom_up_transmission(children[i])

        children_states = torch.cat(children_states, dim=-1)
        children_states = children_states.view(*children_states.shape[:-1], self.max_children, -1)

        if not self.disable_fold:
            self.msg_up[node] = self.fold.add('sNet' + str(0).zfill(3), state, action, children_states, *msg_in)
        else:
            self.msg_up[node] = self.sNet[node](state, action, children_states, *msg_in)

        return self.msg_up[node]

    def top_down_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_td
            else:
                return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

        elif self.msg_down[node] is not None:
            return self.msg_down[node]

        child_state = self.input_state[node]
        children_pos, _  = child_state.split([3, child_state.shape[-1] - 3], dim=-1)
        parent_state = self.input_state[self.parents[node]]
        parent_pos, _  = parent_state.split([3, parent_state.shape[-1] - 3], dim=-1)

        rel_pos = relative_pos_encoding(children_pos, parent_pos)

        # in both-way message-passing, each node takes in its passed-up message as 'state'
        state = self.msg_up[node]
        
        action = self.input_action[node]
        parent_msg = self.top_down_transmission(self.parents[node])

        # find self children index (first child of parent, second child of parent, etc)
        # by finding the number of previous occurences of parent index in the list
        self_children_idx = self.parents[:node].count(self.parents[node])

        # if the structure is flipped, flip message order at the root
        if self.parents[0] == -2 and node == 1:
            self_children_idx = (self.max_children - 1) - self_children_idx

        if not self.disable_fold:
            msg_in = self.fold.add('get_{}'.format(self_children_idx), parent_msg)
        else:
            msg_in = self.msg_slice(parent_msg, self_children_idx)

        if not self.disable_fold:
            self.x1[node], self.x2[node], self.msg_down[node] = self.fold.add('critic' + str(0).zfill(3), rel_pos, state, action, msg_in).split(3)
        else:
            self.x1[node], self.x2[node], self.msg_down[node] = self.critic[node](rel_pos, state, action, msg_in)

        return self.msg_down[node]

    def zero_func_td(self):
        return torch.zeros((1, self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

    def zero_func_bu(self):
        return torch.zeros((1, self.batch_size, self.msg_dim), requires_grad=True).to(device)

    # an ugly way to define functions in a for loop (for torchfold only)
    def addFunction(self, n):
        def f(x):
            return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[n]
        return f

    def msg_slice(self, x, idx):
        return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[idx]

    def clear_buffer(self):
        self.x1 = [None] * self.num_limbs
        self.x2 = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action = [None] * self.num_limbs
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        if not self.disable_fold:
            for i in range(1, self.num_limbs):
                delattr(self, "sNet" + str(i).zfill(3))
        self.parents = parents
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs

        self.sNet = nn.ModuleList([self.sNet[0]] * self.num_limbs)
        if not self.disable_fold:
            for i in range(self.num_limbs):
                setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
