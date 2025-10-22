# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yihan Wang @ University of Oklahoma
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This file is part of the accompanying code to our manuscript:
Y. Wang, L. Zhang, N.B. Erichson, T. Yang. (2025). A Mass Conservation Relaxed (MCR) LSTM Model for Streamflow Simulation
"""


from torch import nn, Tensor
import sys

sys.path.append('../')
import tqdm
import torch.optim as optim


import os
import numpy as np
import torch.nn.functional as F


from typing import Tuple

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import scipy.io as mlio

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

# from tqdm.auto import tqdm
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available


class MassConservingLSTM_MR(nn.Module):
    """ Pytorch implementation of Mass-Conserving LSTMs with Mass Relaxation. """

    def __init__(self, in_dim: int, aux_dim: int, out_dim: int,
                 in_gate: nn.Module = None, out_gate: nn.Module = None,
                 redistribution: nn.Module = None, MR_gate: nn.Module = None,
                 time_dependent: bool = True,
                 batch_first: bool = False):
        """
        Parameters
        ----------
        in_dim : int
            The number of mass inputs.
        aux_dim : int
            The number of auxiliary inputs.
        out_dim : int
            The number of cells or, equivalently, outputs.
        in_gate : nn.Module, optional
            A module computing the (normalised!) input gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `in_dim` x `out_dim` matrix for every sample.
            Defaults to a time-dependent softmax input gate.
        out_gate : nn.Module, optional
            A module computing the output gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` vector for every sample.
        redistribution : nn.Module, optional
            A module computing the redistribution matrix.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` x `out_dim` matrix for every sample.
        MR_gate : nn.Module, optional
            A module implementing the mass relaxing gate (MR gate).
            Defaults to a newly created `_MRGate` instance if not provided.
        batch_first : bool, optional
            Expects first dimension to represent samples if `True`,
            Otherwise, first dimension is expected to represent timesteps (default).
        """
        super().__init__()
        self.in_dim = in_dim
        self.aux_dim = aux_dim
        self.out_dim = out_dim
        self._seq_dim = 1 if batch_first else 0

        gate_inputs = aux_dim + out_dim + in_dim

        self.forward_count = 0
        self.step_count = 0

        # initialize gates
        if out_gate is None:
            self.out_gate = _Gate(in_features=gate_inputs, out_features=out_dim)
        if in_gate is None:
            self.in_gate = _NormalizedGate(in_features=gate_inputs,
                                           out_shape=(in_dim, out_dim),
                                           normalizer="normalized_sigmoid")
        if redistribution is None:
            self.redistribution = _NormalizedGate(in_features=gate_inputs,
                                                  out_shape=(out_dim, out_dim),
                                                  normalizer="normalized_relu")
        '''
        define MR gate
        yhwang Apr 2024
        '''
        # Define MR gate
        if MR_gate is None:
            self.MR_gate = _MRGate(in_features=gate_inputs, out_features=out_dim)

        self._reset_parameters()

    @property
    def batch_first(self) -> bool:
        return self._seq_dim != 0

    def _reset_parameters(self, out_bias: float = -3.):
        nn.init.constant_(self.out_gate.fc.bias, val=out_bias)

    def forward(self, xm, xa, state=None):
        self.forward_count += 1
        xm = xm.unbind(dim=self._seq_dim)
        xa = xa.unbind(dim=self._seq_dim)

        if state is None:
            state = self.init_state(len(xa[0]))

        hs, cs, os, mrs, o_prime_s, mr_flow_s, o_flow_s = [], [], [], [], [], [], []
        for xm_t, xa_t in zip(xm, xa):  # xm, xa [365, 64, 1]
            # xm xa shape: [batchsize, 1] (i.e., [64,1])
            h, state, o, MR, o_prime, MR_flow, o_flow = self._step(xm_t, xa_t,
                                                                   state)  # h.shape=[64,128], state.shape=[64,128]

            hs.append(h)
            cs.append(state)
            os.append(o)
            mrs.append(MR)
            o_prime_s.append(o_prime)
            mr_flow_s.append(MR_flow)
            o_flow_s.append(o_flow)

        hs = torch.stack(hs, dim=self._seq_dim)  # [64, 365, 128]
        cs = torch.stack(cs, dim=self._seq_dim)  # [64, 365, 128]
        os = torch.stack(os, dim=self._seq_dim)
        mrs = torch.stack(mrs, dim=self._seq_dim)
        o_prime_s = torch.stack(o_prime_s, dim=self._seq_dim)
        mr_flow_s = torch.stack(mr_flow_s, dim=self._seq_dim)
        o_flow_s = torch.stack(o_flow_s, dim=self._seq_dim)
        return hs, cs, os, mrs, o_prime_s, mr_flow_s, o_flow_s

    @torch.no_grad()
    def init_state(self, batch_size: int, initial_cell_state=None):  # yhwang May 2024
        """ Create the default initial state. """
        device = next(self.parameters()).device
        if initial_cell_state is None:
            return torch.zeros(batch_size, self.out_dim, device=device)
        else:
            return initial_cell_state

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM_MR. """
        # in this version of the MC-LSTM all available data is used to derive the gate activations. Cell states
        # are L1-normalized so that growing cell states over the sequence don't cause problems in the gates.
        self.step_count += 1
        features = torch.cat([xt_m, xt_a, c / (c.norm(1) + 1e-5)], dim=-1)

        # compute gate activations
        i = self.in_gate(features)
        r = self.redistribution(features)

        o = self.out_gate(features)

        # distribute incoming mass over the cell states
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)

        # reshuffle the mass in the cell states using the redistribution matrix

        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)

        # compute the new mass states
        m_new = m_in + m_sys

        # compute the mass relaxing gate #TODO
        MR = self.MR_gate(c_0_norm=c / (c.norm(1) + 1e-5), f=1 - o)

        o_prime = o + MR
        MR_flow = MR * m_new

        return o * m_new, (1 - o_prime) * m_new, o, MR, o_prime, MR_flow, o * m_new


class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalised gate"""
        return torch.sigmoid(self.fc(x))


class _MRGate(nn.Module):
    """Utility class to implement a mass relaxing gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_MRGate, self).__init__()
        # Declare learnable parameters
        self.bias_b0_yrm = nn.Parameter(torch.FloatTensor(out_features))
        self.weight_s_yvm = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_r_yvm = nn.Parameter(torch.FloatTensor(out_features, in_features))

        # Initialize the learnable parameters
        self._reset_parameters()
        # Define a ReLU activation function
        self.relu_v = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_features)

    def _reset_parameters(self):
        # Initialize learnable parameters using specific initialization methods
        nn.init.orthogonal_(self.weight_s_yvm)  # Initialize weight_s_yvm orthogonally
        nn.init.orthogonal_(self.weight_r_yvm)  # Initialize weight_r_yvm orthogonally
        nn.init.zeros_(self.bias_b0_yrm)  # Initialize bias_b0_yrm to zeros

    def forward(self, c_0_norm: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the Mass Relaxing gate."""

        # Compute ov0 using matrix multiplication and exponential function
        ov0 = torch.mm(c_0_norm - self.bias_b0_yrm, torch.exp(self.weight_s_yvm))
        # ov0 shape: [batch size, out_features]
        # f shape: [batch size, out_features]
        # weight_r_yvm shape: [out_features, in_features]
        # weight_s_yvm shape: [out_features, in_features]
        # Compute ov1 using sigmoid and tan activations
        '''
        **** c_0_norm shape:  torch.Size([256, 64]) bias_b0_yrm shape:  torch.Size([64]) weight_s_yvm shape:  torch.Size([64, 96])
        **** ov0 shape:  torch.Size([256, 96]) f shape:  torch.Size([256, 96]) weight_r_yvm shape:  torch.Size([64, 96])
        '''
        ov1 = self.layer_norm(torch.tanh(ov0) @ torch.sigmoid(self.weight_r_yvm.t()))
        # Compute the final ov value using ReLU activation
        ov2 = ov1 - self.relu_v(ov1 - f)  # ensure 1-o-MR>=0
        ov = self.relu_v(ov2 + 1 - f) + f - 1  # ensure o+MR>=0 #yihanwang test commented this out

        return ov


class _NormalizedGate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown normalizer {normalizer}. Must be one of {'normalized_sigmoid', 'normalized_relu'}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalized gate"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.
    :param model: A torch.nn.Module implementing the MC-LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    loss_list = []
    model.train()
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)

        xm = xs[..., 0:1]
        xa = xs[..., 1:]
        # get model predictions
        m_out, c, o, mr, o_prime, mr_flow, o_flow = model(xm, xa)  # [batch size, seq length, hidden size]

        # y_hat = m_out[:, -1:].sum(dim=-1)
        output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
        # y_hat = output.transpose(0, 1)
        y_hat = output[:, -1, :]
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        loss_list.append(loss)
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
    loss_ave = np.mean(torch.stack(loss_list).detach().cpu().numpy())
    return loss_ave


def eval_model(model, loader) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    hidden = []
    cell = []
    MR_flow_all = []
    out_flow_all = []

    with torch.no_grad():
        COUNT = 0
        # request mini-batch of data from the loader
        for xs, ys in loader:
            COUNT += 1
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            xm = xs[..., 0:1]
            xa = xs[..., 1:]
            # get model predictions
            m_out, c, o, mr, o_prime, mr_flow, o_flow = model(xm, xa)

            output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
            y_hat = output[:, -1, :]
            hidden_state = m_out[:, -1, :]  # [batch size, 1, hidden sizes]
            cell_state = c[:, -1, :].sum(dim=-1, keepdim=True)

            MR_flow = mr_flow[:, -1, :].sum(dim=-1, keepdim=True)
            out_flow = o_flow[:, -1, :].sum(dim=-1, keepdim=True)

            obs.append(ys)
            preds.append(y_hat)
            hidden.append(hidden_state)
            cell.append(cell_state)

            MR_flow_all.append(MR_flow)
            out_flow_all.append(out_flow)
    return torch.cat(obs), torch.cat(preds), torch.cat(hidden), torch.cat(cell), torch.cat(MR_flow_all), torch.cat(
        out_flow_all)  # torch.cat(out), torch.cat(MR), torch.cat(out_prime)


"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""
class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, cfr, cfi, N=64, dt_min=0.0001, dt_max=0.1, lr=None, lr_dt=None, wd=None):
        super().__init__()
        # Generate dt

        H = d_model
        log_dt = torch.rand(H) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, 0, lr=lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2)) * cfr
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H) * cfi
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(
            -1)  # (H N) discretizing the continuous-time dynamics to generate a discrete-time convolution kernel
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, wd, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": wd}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class LTI(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, add_noise=0, mult_noise=0, cfr=1, cfi=1, transposed=True,
                 **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.add_noise = add_noise
        self.mult_noise = mult_noise

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, cfr, cfi, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        ybar = u_f * k_f
        y = torch.fft.irfft(ybar, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed: y = y.transpose(-1, -2)
        return y, None  # Return a dummy state to satisfy this repo's interface, but this can be modified

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            add_noise = add_noise_level * np.random.beta(2, 5) * torch.cuda.FloatTensor(x.shape).normal_()
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * np.random.beta(2, 5) * (
                        2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1) + 1
    return mult_noise * x + add_noise


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


def warmup_cosine_annealing_lr(epoch, warmup_epochs, total_epochs, lr_start=1.0, lr_end=0.05):
    if epoch < warmup_epochs:
        lr = lr_start * (epoch / warmup_epochs)
    else:
        lr = lr_end + (lr_start - lr_end) * 0.5 * (
                1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )
    return lr


class HOPE(nn.Module):

    def __init__(
            self,
            d_input,
            d_output=1,
            d_model=256,
            n_layers=4,
            dropout=0.1,
            cfg=None,
            prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                LTI(d_model, dropout=dropout, transposed=True,
                    lr=min(cfg["lr_min"], cfg["lr"]), d_state=cfg["d_state"], dt_min=cfg["min_dt"],
                    dt_max=cfg["max_dt"], lr_dt=cfg["lr_dt"], cfr=cfg["cfr"], cfi=cfg["cfi"], wd=cfg["wd"])
            )
            # self.norms.append(nn.LayerNorm(d_model))
            self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

        # self.lnorm = torch.nn.LayerNorm(365)
        # self.att = SelfAttentionLayer(365) #20240626 remove attention

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # x = torch.cat((x, torch.zeros((x.shape[0],1,x.shape[2])).to('cuda')), 1 )
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        # x = torch.cat((x,torch.flip(x,dims=[-1])),dim=-1) # bi-directional

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x

            # z, _ = self.att(z)

            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


def setup_optimizer(model, lr, weight_decay, epochs, warmup_epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Define lambda function for the scheduler
    lr_lambda = lambda epoch: warmup_cosine_annealing_lr(epoch, warmup_epochs, epochs, lr_start=1.0, lr_end=0.0)

    # Create the scheduler with LambdaLR
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
                             f"Optimizer group {i}",
                             f"{len(g['params'])} tensors",
                         ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler



class LSTM(nn.Module):
    """Implementation of the standard LSTM.
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data = weight_hh_data
        nn.init.constant_(self.bias.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]

        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.

        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        h_0 = x.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x[t], self.weight_ih))
            f, i, o, g = gates.chunk(4, 1)

            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n


class LSTM_Model(nn.Module):
    """Wrapper class that connects LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(LSTM_Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        self.lstm = LSTM(
            input_size=input_size_dyn,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        h_n, c_n = self.lstm(x_d)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n
