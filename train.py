import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
import zarr
import numpy as np
import tqdm

# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst, dtype=torch.float)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float)

        # r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, action_dim)

        self.lr = q_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


class DQNAgent:
    def __init__(self):
        self.state_dim     = 3
        self.action_dim    = 9  
        self.lr            = 0.01
        self.gamma         = 0.98
        self.tau           = 0.01
        self.epsilon       = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min   = 0.001
        self.buffer_size   = 100000
        self.batch_size    = 200
        self.memory        = ReplayBuffer(self.buffer_size)

        self.Q        = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, state):
        random_number = np.random.rand()
        maxQ_action_count = 0
        if self.epsilon < random_number:
            with torch.no_grad():
                action = float(torch.argmax(self.Q(state)).numpy())
                # action = float(action.numpy())
                real_action = (action - 4) / 4
                maxQ_action_count = 1
        else:
            action = np.random.choice([n for n in range(9)])
            real_action = (action - 4) / 2  

        return action, real_action, maxQ_action_count

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            q_target = self.Q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * done * q_target
        return target

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch
        a_batch = a_batch.type(torch.int64)

        td_target = self.calc_target(mini_batch)

        #### Q train ####
        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.smooth_l1_loss(Q_a, td_target)
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()
        self.Q.optimizer.step()
        #### Q train ####

        #### Q soft-update ####
        for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


class DQNAgent:
    def __init__(self, wight_file_path):
        self.state_dim     = 3
        self.action_dim    = 9
        self.lr            = 0.002
        self.trained_model = wight_file_path

        self.Q        = QNetwork(self.state_dim, self.action_dim, self.lr)
        self.Q.load_state_dict(torch.load(self.trained_model))

    def choose_action(self, state):
        with torch.no_grad():
            action = float(torch.argmax(self.Q(state)).numpy())
            real_action = (action - 4) / 2
        return real_action

#agent.choose_action(torch.FloatTensor(state))

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
class PendulumImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):
        dataset_root = zarr.open(dataset_path, 'r')
        train_image_data = dataset_root['images'][:]
        train_image_data = np.moveaxis(train_image_data, -1, 1)
        # (N,3,96,96)

        train_data = {
            'pendulum_state': dataset_root['states'][:],
            'action': dataset_root['actions'][:]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        nsample['image'] = nsample['image'][:self.obs_horizon, :]
        nsample['pendulum_state'] = nsample['pendulum_state'][:self.obs_horizon, :]
        return nsample


dataset_path = '/n/home04/amuppidi/DiffusionPendulum/pendulum_data.zarr'
pred_horizon = 16
obs_horizon = 8
action_horizon = 1

dataset = PendulumImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)


# %%


# %%
sample = dataset[0]
sample

# %%
# save training data statistics (min, max) for each dim
stats = dataset.stats

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['image'].shape:", batch['image'].shape)
print("batch['agent_pos'].shape:", batch['pendulum_state'].shape)
print("batch['action'].shape", batch['action'].shape)

# %%
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

## Diffusion classes edited and taken directly 
# from https://github.com/real-stanford/diffusion_policy/tree/main/diffusion_policy/model

#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


vision_encoder = get_resnet('resnet18')
pred_horizon = 16
obs_horizon = 8
action_horizon = 1
vision_encoder = replace_bn_with_gn(vision_encoder)
vision_feature_dim = 512
lowdim_obs_dim = 3
obs_dim = vision_feature_dim + 3 #+ lowdim_obs_dim
action_dim = 1

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

num_diffusion_iters = 100
device = torch.device('cuda')
_ = nets.to(device)

# %%
from tqdm import tqdm
num_epochs = 75
ema = EMAModel(
    parameters=nets.parameters(),
    model=nets,
    power=0.75)

optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    for epoch_idx in tglobal:
        epoch_loss = list()
        for nbatch in dataloader:

            nimage = nbatch['image'][:,:obs_horizon].to(device)
            nagent_pos = nbatch['pendulum_state'][:,:obs_horizon].to(device)
            naction = nbatch['action'].to(device)
            B = nagent_pos.shape[0]

            image_features = nets['vision_encoder'](
                nimage.flatten(end_dim=1))
            image_features = image_features.reshape(
                *nimage.shape[:2],-1)
            # (B,obs_horizon,D)
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, obs_horizon * obs_dim)
            #print("obs_cond.shape " + str(obs_cond.shape))
            #print("obs_horizon * obs_dim " + str(obs_horizon * obs_dim))
            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            ema.step(nets)

            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# is used for inference
#ema_nets = ema.averaged_model

# %%
torch.save(ema.state_dict(), 'new_data_30_ema_nets_model_weights.pth')
torch.save(nets.state_dict(), 'new_data_30_nets_model_weights.pth')


# %%
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})
ema = EMAModel(
    parameters=nets.parameters(),
    model=nets,
    power=0.75)

# %%
# Load the state dictionaries
ema_state_dict = torch.load('new_data_30_ema_nets_model_weights.pth')
nets_state_dict = torch.load('new_data_30_nets_model_weights.pth')

# Update the initial model with the saved state dictionaries
ema.load_state_dict(ema_state_dict)
nets.load_state_dict(nets_state_dict)


# %%
ema.copy_to(nets.parameters())

ema_nets = nets 
# %%
max_steps = 100
env = gym.make('Pendulum-v1', new_step_api=True)

# Set eval seeds here! Or set up eval training loop
env.seed(100000)

r = env.reset()
r

obs = r
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

# %%
#from skvideo.io import vwrite
with tqdm(total=max_steps, desc="Eval Pend") as pbar:
    while not done:
        B = 1
        images = []
        agent_poses = []

        for x in obs_deque:
            img = env.render(mode='rgb_array')
            images.append(img)

            # Extract the sin and cos components from the observation
            agent_pos = x 
            agent_poses.append(agent_pos)

        images = np.stack(images)
        agent_poses = np.stack(agent_poses)
        nagent_poses = normalize_data(agent_poses, stats=stats['pendulum_state'])
        nimages = images
        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
        nimages = nimages.permute(0, 3, 1, 2)

        # (2,3,96,96)
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
        # (2,2)

        with torch.no_grad():
            image_features = ema_nets['vision_encoder'](nimages)
            # (2,512)

            obs_features = torch.cat([image_features, nagent_poses], dim=-1)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        start = obs_horizon - 1
        end = start + 1
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        for i in range(len(action)):
            obs, reward, done, _, info = env.step(action[i])
            obs_deque.append(obs)
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

print('Score: ', max(rewards))

# visualize
#from IPython.display import Video
#vwrite('vis.mp4', imgs)
#Video('vis.mp4', embed=True, width=256, height=256)

# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def visualize_images_as_animation(imgs):
    fig, ax = plt.subplots()
    ax.set_title("Pendulum Animation")
    ax.axis('off')
    img_plot = ax.imshow(imgs[0])

    def update(frame):
        img_plot.set_data(frame)
        return [img_plot]

    ani = animation.FuncAnimation(fig, update, frames=imgs, blit=True)
    return HTML(ani.to_html5_video())
visualize_images_as_animation(imgs)

# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def save_images_as_video(imgs, file_path):
    fig, ax = plt.subplots()
    #ax.set_title("Pendulum Animation")
    ax.axis('off')
    img_plot = ax.imshow(imgs[0])

    def update(frame):
        img_plot.set_data(frame)
        return [img_plot]

    ani = animation.FuncAnimation(fig, update, frames=imgs, blit=True)

    ani.save(file_path, writer='ffmpeg', fps=30)
    plt.close(fig)
save_images_as_video(imgs, 'pendulum_animation.mp4')

# %%
