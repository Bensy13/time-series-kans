import torch
import torch.nn as nn
from kan import KAN

class KANEncoder(nn.Module):
    def __init__(
        self,
        width: list,
        grid: int = 21,
        k: int = 3,
        mult_arity: int = 2, # keep as int, same arity per layer if using mutiplicative nodes
        base_fun: str = 'silu', # identity an option
        symbolic_enabled: bool = True,
        affine_trainable: bool = False, # sticks to KA theorem more closely
        grid_range: list = [0, 1.0], # when using grid update not oto important.
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        seed: int = 1,
        sparse_init = False,
        save_act: bool = True,
        first_init: bool = True,
        device: str = 'cuda',
        auto_save = False,
        ckpt_path='./kan_model'
    ):
        super(KANEncoder, self).__init__()

        self.kan = KAN(
            width=width,
            grid=grid,
            k=k,
            mult_arity=mult_arity,
            base_fun=base_fun,
            symbolic_enabled=symbolic_enabled,
            affine_trainable=affine_trainable,
            grid_range=grid_range,
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            seed=seed,
            sparse_init=sparse_init,
            save_act=save_act,
            auto_save=auto_save,
            first_init=first_init,
            device=device,
            ckpt_path=ckpt_path
        )

    def forward(self, x):
        # x: [B, T, F] — batch, time, features
        B, T, F = x.shape
        x_flat = x.view(B * T, F)         # Flatten time
        out_flat = self.kan(x_flat)       # [B*T, latent_dim]
        return out_flat.view(B, T, -1)    # Reshape back to [B, T, latent_dim]

    def update_grid(self, x):
        # x: [B, T, F] — batch, time, features
        B, T, F = x.shape
        x_flat = x.view(B * T, F)         # Flatten time
        return self.kan.update_grid(x_flat)       # [B*T, latent_dim]


    """
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_epsilon(how close to grid - (a+eps, b-eps, k)=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1, save_act=True, sparse_init=False, auto_save=True, first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu'):
        '''
        initalize a KAN model
        
        Args:
        -----
            width : list of int
                Without multiplication nodes: :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
                With multiplication nodes: :math:`[[n_0,m_0=0], [n_1,m_1], .., [n_{L-1},m_{L-1}]]` specify the number of addition/multiplication nodes in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
            noise_scale : float
                initial injected noise to spline.
            base_fun : str
                the residual function b(x). Default: 'silu'
            symbolic_enabled : bool
                compute (True) or skip (False) symbolic computations (for efficiency). By default: True. 
            affine_trainable : bool
                affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default updata_grid=True)
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            save_act : bool
                indicate whether intermediate activations are saved in forward pass
            sparse_init : bool
                sparse initialization (True) or normal dense initialization. Default: False.
            auto_save : bool
                indicate whether to automatically save a checkpoint once the model is modified
            state_id : int
                the state of the model (used to save checkpoint)
            ckpt_path : str
                the folder to store checkpoints. Default: './model'
            round : int
                the number of times rewind() has been called
            device : str
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        checkpoint directory created: ./model
        saving model version 0.0
    """