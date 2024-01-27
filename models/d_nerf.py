import sys
sys.path.append('/workspaces/CV-NeRF-Task3-Deformation')

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import PositionalEncoding

class original_NeRF(nn.Module):
    """
    The neural network for NeRF.
    
    This class is ported from kwea123/nerf_pl 
    (https://github.com/kwea123/nerf_pl/tree/master)
    under the MIT license.
    """
    def __init__(self,
                 D=8, 
                 W=256,
                 in_channels_xyz=60, 
                 in_channels_dir=24,
                 skips=[4],
                 ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3*10*2=60 by default)
        in_channels_dir: number of input channels for direction (3*4*2=24 by default)
        skips: add skip connection in the Dth layer
        """
        super(original_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Sequential(nn.Linear(W, 1),
                                   nn.Sigmoid())
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())
        

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out

class D_NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 in_channels_xyz=3, 
                 in_channels_dir=3,
                 in_channels_time=1,
                 sample_num=0,
                 skips=[4],
                 xyz_L=10,
                 dir_L=4,
                 ):
        super(D_NeRF, self).__init__()
        self.D=D
        self.W=W
        self.in_channels_xyz=in_channels_xyz
        self.in_channels_dir=in_channels_dir
        self.in_channels_time=in_channels_time
        self.skips=skips
        self.xyz_L=xyz_L
        self.dir_L=dir_L
        self.sample_num = sample_num
        self.canonical_net=original_NeRF(D,W,
                                         in_channels_xyz*2*xyz_L,
                                         in_channels_dir*2*dir_L,
                                         skips)

        self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.in_channels_xyz + self.in_channels_time, self.W)]
        for i in range(self.D - 1):
            layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.in_channels_xyz

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts):
        '''
        Inputs:
            ...
            ts: time (B,1)

        Outputs:
            ...
            dx: relative displacement
        '''
        xyzs, dirs = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        # time layer, calculate dx to original scene
        cur_time = ts[0,0]
        if cur_time == 0. :
            dx = torch.zeros_like(xyzs[:, :3])
        else:
            dx = self.query_time(xyzs, ts, self._time, self._time_out)

        # encoding pts and views
        xyz_L = int(self.xyz_L)
        dir_L = int(self.dir_L)
        xyz_encoder = PositionalEncoding(xyz_L)
        xyz_encoded = xyz_encoder(xyzs + dx)	# (ray_num * sample_num) * (6 * xyz_L)
        dir_encoder = PositionalEncoding(dir_L)
        dir_encoded = dir_encoder(dirs) # ray_num * (6 * dir_L)
        dir_encoded = torch.repeat_interleave(dir_encoded, self.sample_num, dim=0) # (ray_num * sample_num) * (6 * dir_L)
        xyz_dir_encoded = torch.cat((xyz_encoded, dir_encoded), dim=1)

        # canonical layer
        out = self.canonical_net(xyz_dir_encoded)
        return out, dx

# model=D_NeRF()
# print(model)