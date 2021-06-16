from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MLPBase
import torchfold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
class RelPositionEncoding(nn.Module):
    def __init__(self, n_hid, max_len = 241, dropout = 0.2):
        super(RelPositionEncoding, self).__init__()
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))

def relative_pos_encoding(children_pos, parent_pos):
    neighbor_xyz = self.gather_neighbour(pos, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = torch.abs(relative_xyz)
    relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
    return relative_feature    

    d_in = feature.get_shape()[-1].value
    f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
    f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
    f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
    f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
    f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)
'''
class ActorUp(nn.Module):
    """a bottom-up module used in bothway message passing that only passes message to its parent"""
    def __init__(self, state_dim, msg_dim, max_children):
        super(ActorUp, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)

    def forward(self, x, *m):
        m = torch.cat(m, dim=-1)
        x = self.fc1(x)
        x = F.normalize(x, dim=-1)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        xm = self.fc2(xm)
        xm = torch.tanh(xm)
        xm = self.fc3(xm)
        xm = F.normalize(xm, dim=-1)
        msg_up = xm

        return msg_up


class ActorDownAction(nn.Module):
    """a top-down module used in bothway message passing that passes messages to children and outputs action"""
    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, self_input_dim, action_dim, msg_dim, max_action, max_children):
        super(ActorDownAction, self).__init__()
        self.max_action = max_action
        self.action_base = MLPBase(self_input_dim + msg_dim, action_dim)
        self.msg_base = MLPBase(self_input_dim + msg_dim, msg_dim * max_children)

    def forward(self, x, m):
        xm = torch.cat((x, m), dim=-1)
        xm = torch.tanh(xm)
        action = self.max_action * torch.tanh(self.action_base(xm))
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)
        return action, msg_down

'''
    parents: parent indices of pre-ordered limbs, thus children will have the same index.
'''

class ActorGraphPolicy(nn.Module):
    """a weight-sharing dynamic graph policy that changes its structure based on different morphologies and passes messages between nodes"""
    def __init__(self, state_dim, action_dim, msg_dim, batch_size, max_action, max_children, disable_fold):
        super(ActorGraphPolicy, self).__init__()
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim

        assert self.action_dim == 1
        
        # The same ActorUp network got shallow-copied
        self.sNet = nn.ModuleList([ActorUp(state_dim, msg_dim, max_children)] * self.num_limbs).to(device)
        if not self.disable_fold:
            for i in range(self.num_limbs):
                setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        
        # The same ActorDownAction network got shallow-copied
        # we pass msg_dim as first argument because in both-way message-passing, each node takes in its passed-up message as 'state'
        self.actor = nn.ModuleList([ActorDownAction(msg_dim, action_dim, msg_dim, max_action, max_children)] * self.num_limbs).to(device)
        if not self.disable_fold:
            for i in range(self.num_limbs):
                setattr(self, "actor" + str(i).zfill(3), self.actor[i])

        if not self.disable_fold:
            for i in range(self.max_children):
                setattr(self, 'get_{}'.format(i), self.addFunction(i))

    def forward(self, state, mode='train'):
        self.clear_buffer()
        if mode == 'inference':
            temp = self.batch_size
            self.batch_size = 1
        if not self.disable_fold:
            self.fold = torchfold.Fold()
            self.fold.cuda()
            self.zeroFold_td = self.fold.add("zero_func_td")
            self.zeroFold_bu = self.fold.add("zero_func_bu")
            self.a = []
        assert state.shape[1] == self.state_dim * self.num_limbs, 'state.shape[1] expects {} but got {} with num_limbs being {} and state_dim being {}'.format(self.state_dim * self.num_limbs, state.shape[1], self.num_limbs, self.state_dim)

        for i in range(self.num_limbs):
            self.input_state[i] = state[:, i * self.state_dim:(i + 1) * self.state_dim]
            if not self.disable_fold:
                self.input_state[i] = torch.unsqueeze(self.input_state[i], 0)

        # bottom up transmission by recursion
        for i in range(self.num_limbs):
            self.bottom_up_transmission(i)

        # top down transmission by recursion
        for i in range(self.num_limbs):
            self.top_down_transmission(i)


        if not self.disable_fold:
            self.a += self.action
            self.action = self.fold.apply(self, [self.a])[0]
            self.action = torch.transpose(self.action, 0, 1)
            self.fold = None
        else:
            self.action = torch.stack(self.action, dim=-1)
            self.msg_down = torch.stack(self.msg_down, dim=-1)

        if mode == 'inference':
            self.batch_size = temp

        return torch.squeeze(self.action)

    def bottom_up_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_bu
            else:
                return torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device)

        if self.msg_up[node] is not None:
            return self.msg_up[node]

        state = self.input_state[node]

        # children indices --> not efficient
        children = [i for i, x in enumerate(self.parents) if x == node]
        assert (self.max_children - len(children)) >= 0
        children += [-1] * (self.max_children - len(children))
        msg_in = [None] * self.max_children
        for i in range(self.max_children):
            msg_in[i] = self.bottom_up_transmission(children[i]) # children[i] = children node idx

        if not self.disable_fold:
            self.msg_up[node] = self.fold.add('sNet' + str(0).zfill(3), state, *msg_in)
        else:
            self.msg_up[node] = self.sNet[node](state, *msg_in)

        return self.msg_up[node]

    def top_down_transmission(self, node):
        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_td
            else:
                return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

        elif self.msg_down[node] is not None:
            return self.msg_down[node]

        # in both-way message-passing, each node takes in its passed-up message as 'state'
        state = self.msg_up[node]
            
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
            self.action[node], self.msg_down[node] = self.fold.add('actor' + str(0).zfill(3), state, msg_in).split(2)
        else:
            self.action[node], self.msg_down[node] = self.actor[node](state, msg_in)

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
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
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
