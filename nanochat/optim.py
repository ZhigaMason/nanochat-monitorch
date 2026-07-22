"""
A highly efficient Multi-Optimizer combining Muon, standard AdamW, and Coupled AdamW.
- 'muon': For 2D matrix parameters (Transformer blocks, MLPs).
- 'coupled_adamw': For anisotropic-prone 2D Embeddings (wte, ve).
- 'adamw': For general scalars, biases, and final LayerNorms.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
# FUSED KERNELS
# -----------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor,
    step_t: Tensor, lr: float, beta1: float, beta2: float,
    eps: float, wd: float,
) -> None:
    """Standard AdamW Fused Kernel"""
    # 1. Cast grad to parameter dtype for safety
    grad_cast = grad.to(p.dtype)
    
    p.mul_(1.0 - lr * wd)
    exp_avg.lerp_(grad_cast, 1.0 - beta1)
    exp_avg_sq.lerp_(grad_cast.square(), 1.0 - beta2)
    
    bias1 = 1.0 - beta1 ** step_t
    bias2 = 1.0 - beta2 ** step_t
    
    denom = (exp_avg_sq / bias2).sqrt() + eps
    step_size = lr / bias1
    
    update = (exp_avg / denom) * step_size
    p.sub_(update)


@torch.compile(dynamic=False, fullgraph=True)
def coupled_adamw_step_fused(
    p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor,
    step_t: Tensor, lr: float, beta1: float, beta2: float,
    eps: float, wd: float, avg_scale: float
) -> None:
    """Coupled AdamW Fused Kernel"""
    grad_cast = grad.to(p.dtype)

    p.mul_(1.0 - lr * wd)
    
    exp_avg.lerp_(grad_cast, 1.0 - beta1)
    exp_avg_sq.lerp_(grad_cast.square(), 1.0 - beta2)
    
    coupled_v = exp_avg_sq.mean(dim=0, keepdim=True) * avg_scale
    
    bias1 = 1.0 - (beta1 ** step_t)
    bias2 = 1.0 - (beta2 ** step_t)
    
    denom = (coupled_v / bias2).sqrt() + eps
    step_size = lr / bias1
    
    update = (exp_avg / denom) * step_size
    p.sub_(update)


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor, stacked_params: Tensor, momentum_buffer: Tensor,
    second_momentum_buffer: Tensor, momentum: float, lr: float,
    wd: float, beta2: float, ns_steps: int, red_dim: int,
) -> None:
    """Muon Orthogonalization Fused Kernel"""
    momentum_buffer.lerp_(stacked_grads, 1.0 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1.0 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    mask = (g * stacked_params) >= 0
    
    update = lr * g + lr * wd * stacked_params * mask
    stacked_params.sub_(update)


# -----------------------------------------------------------------------------
# SINGLE GPU MULTI-OPTIMIZER
# -----------------------------------------------------------------------------

class MultiOptimizer(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # Removed the self._*_t CPU tensors that caused the glibc tcache crash

    def _step_adam_variant(self, group: dict, is_coupled: bool) -> None:
        for p in group['params']:
            if p.grad is None: 
                continue
            grad = p.grad
            state = self.state[p]

            if not state:
                # Store step as a persistent device tensor to avoid Python GC race condition
                state['step_t'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                
            state['step_t'] += 1.0

            # Extract statically to Python floats to pass to compiled kernel
            lr = float(group['lr'])
            beta1 = float(group['betas'][0])
            beta2 = float(group['betas'][1])
            eps = float(group['eps'])
            wd = float(group['weight_decay'])
            
            if is_coupled:
                avg_scale = float(group.get('avg_scale', 1.0))
                coupled_adamw_step_fused(
                    p, grad, state['exp_avg'], state['exp_avg_sq'],
                    state['step_t'], lr, beta1, beta2, eps, wd, avg_scale
                )
            else:
                adamw_step_fused(
                    p, grad, state['exp_avg'], state['exp_avg_sq'],
                    state['step_t'], lr, beta1, beta2, eps, wd
                )

    def _step_muon(self, group: dict) -> None:
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Pass pure floats to avoid concurrency issues with the C++ backend
        momentum = float(group["momentum"])
        beta2 = float(group["beta2"] if group["beta2"] is not None else 0.0)
        lr = float(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        wd = float(group["weight_decay"])

        muon_step_fused(
            stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
            momentum, lr, wd, beta2, group["ns_steps"], red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            kind = group['kind']
            if kind == 'adamw':
                self._step_adam_variant(group, is_coupled=False)
            elif kind == 'coupled_adamw':
                self._step_adam_variant(group, is_coupled=True)
            elif kind == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")


# -----------------------------------------------------------------------------
# DISTRIBUTED MULTI-OPTIMIZER
# -----------------------------------------------------------------------------

class DistMultiOptimizer(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # Removed the self._*_t CPU tensors that caused the glibc tcache crash

    def _reduce_adam_variant(self, group: dict, world_size: int) -> dict:
        param_infos = {}
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] divisible by world_size"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            return dict(future=None, params=params)

        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size, params=params)

    def _compute_adam_variant(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int, is_coupled: bool) -> None:
        param_infos = info['param_infos']
        for p in group['params']:
            if p.grad is None:
                continue
            
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            if not state:
                state['step_t'] = torch.tensor(0.0, dtype=torch.float32, device=p_slice.device)
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
                
            state['step_t'] += 1.0

            lr = float(group['lr'])
            beta1 = float(group['betas'][0])
            beta2 = float(group['betas'][1])
            eps = float(group['eps'])
            wd = float(group['weight_decay'])
            
            if is_coupled:
                avg_scale = float(group.get('avg_scale', 1.0))
                coupled_adamw_step_fused(
                    p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                    state['step_t'], lr, beta1, beta2, eps, wd, avg_scale
                )
            else:
                adamw_step_fused(
                    p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                    state['step_t'], lr, beta1, beta2, eps, wd
                )

            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None, hold_buffer=p_slice))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        if info['future'] is None:
            return
            
        info['future'].wait()
        params = info['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            momentum = float(group["momentum"])
            beta2 = float(group["beta2"])
            lr = float(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            wd = float(group["weight_decay"])
            
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                momentum, lr, wd, beta2, group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params, hold_buffer=updated_params))

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos: list[dict] = []
        for group in self.param_groups:
            kind = group['kind']
            if kind in ('adamw', 'coupled_adamw'):
                reduce_infos.append(self._reduce_adam_variant(group, world_size))
            elif kind == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {kind}")

        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            kind = group['kind']
            if kind == 'adamw':
                self._compute_adam_variant(group, info, gather_list, rank, world_size, is_coupled=False)
            elif kind == 'coupled_adamw':
                self._compute_adam_variant(group, info, gather_list, rank, world_size, is_coupled=True)
            elif kind == 'muon':
                self._compute_muon(group, info, gather_list, rank)

        self._finish_gathers(gather_list)
