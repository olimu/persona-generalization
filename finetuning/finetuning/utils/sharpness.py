# sharpness.py
"""Utility functions for computing sharpness metrics (Hessian eigenvalues, traces, etc.)"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, List, Dict, Optional, Tuple
from tqdm import tqdm
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._expm_multiply import traceest
from torch.nn.utils import parameters_to_vector
from contextlib import nullcontext

# Prefer the new sdpa_kernel API for higher-order grads; fall back if unavailable.
def _sdpa_math_ctx():
    """
    Returns a context manager that forces the SDPA backend to MATH (no Flash/Efficient)
    so second derivatives (double backward) are supported. Falls back gracefully.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch ≥ 2.3
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        # Older API (deprecated but still works on many installs)
        try:
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            return nullcontext()


def prepare_batches_for_sharpness(
    eval_data: Dict,
    tokenizer,
    max_length: int,
    batch_size: int,
    include_labels: bool = False
) -> List[Dict[str, torch.Tensor]]:
    """
    Prepare batches for sharpness computation.

    Args:
        eval_data: Dictionary containing 'input_ids', 'attention_mask', and optionally 'labels'
        tokenizer: Tokenizer with pad_token_id
        max_length: Maximum sequence length
        batch_size: Batch size
        include_labels: Whether to include labels in the batches

    Returns:
        List of batch dictionaries with 'input_ids', 'attention_mask', and optionally 'labels'
    """
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    batches = []
    for i in range(0, len(eval_data['input_ids']), batch_size):
        batch_seqs = eval_data['input_ids'][i:i+batch_size]
        batch_max_length = min(max_length, max(len(x) for x in batch_seqs))

        input_ids = torch.stack([
            torch.tensor([pad_token] * (batch_max_length - len(x)) + x[:batch_max_length])
            for x in batch_seqs
        ])

        attention_mask = torch.stack([
            torch.tensor([0] * (batch_max_length - len(x)) + [1] * len(x[:batch_max_length]))
            for x in batch_seqs
        ])

        batch_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        if include_labels:
            labels = torch.stack([
                torch.tensor([-100] * (batch_max_length - len(x)) + eval_data['labels'][i+j][:batch_max_length])
                for j, x in enumerate(batch_seqs)
            ])
            batch_dict['labels'] = labels

        batches.append(batch_dict)

    return batches


def create_causal_lm_loss_fn(reduction: str = 'mean'):
    """
    Create a standard causal language modeling loss function.

    Args:
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss function that takes (logits, target) and returns loss
    """
    def loss_fn(logits, target):
        logits = logits[:, :-1, :].contiguous()
        target = target[:, 1:].contiguous()
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=-100,
            reduction=reduction
        )
    return loss_fn


@torch.no_grad()
def _flatten_num_params(params: List[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def _call_loss_fn_flex(
    model,
    loss_fn: Callable,
    batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Calls user-provided loss_fn with either signature:
      - loss_fn(model, batch)
      - loss_fn(logits, target)
    """
    import inspect
    sig = inspect.signature(loss_fn)
    if len(sig.parameters) == 2:
        first = next(iter(sig.parameters.keys()))
        if first == 'model' or 'model' in first.lower():
            return loss_fn(model, batch)
        else:
            out = model(**batch)
            logits = out.logits if hasattr(out, "logits") else out
            return loss_fn(logits, batch.get('labels', batch['input_ids']))
    else:
        out = model(**batch)
        logits = out.logits if hasattr(out, "logits") else out
        return loss_fn(logits, batch.get('labels', batch['input_ids']))


def compute_hvp(
    model,
    loss_fn: Callable,
    batches: List[Dict[str, torch.Tensor]],
    vector: torch.Tensor,
    normalize_by_tokens: bool = False,
    progress_desc: str = 'Computing HVP'
) -> torch.Tensor:
    """
    Compute Hessian-vector product (HVP) using Pearlmutter's trick, fully on GPU.
    Fresh SDPA + autocast contexts are created per-iteration (generator ctxmgrs are single-use).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    p_dim = sum(p.numel() for p in params)

    v = vector.reshape(-1)
    if v.numel() != p_dim:
        raise ValueError(f"Vector length {v.numel()} does not match #trainable params {p_dim}.")

    device = next(model.parameters()).device
    v = v.to(device, dtype=torch.float32)
    hvp = torch.zeros(p_dim, device=device, dtype=torch.float32)

    total_tokens = 0
    total_examples = 0

    for batch in batches:
        batch = {k: (t.to(device, non_blocking=True) if torch.is_tensor(t) else t) for k, t in batch.items()}

        model.zero_grad(set_to_none=True)
        model.train(False)  # eval-mode for deterministic Hessian

        with torch.cuda.amp.autocast(enabled=False):
            with _sdpa_math_ctx(): 
                loss = _call_loss_fn_flex(model, loss_fn, batch)

        # First-order grads with graph for HVP
        grads = torch.autograd.grad(loss, inputs=params, create_graph=True, retain_graph=True)
        g_flat = parameters_to_vector(grads)

        # Dot with v and second backward => H v
        dot = (g_flat * v).sum()
        hv_tensors = torch.autograd.grad(dot, params, retain_graph=False)
        hv = parameters_to_vector(hv_tensors).detach()
        hvp += hv

        # Normalization bookkeeping
        bs = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
        total_examples += bs
        if normalize_by_tokens:
            if 'labels' in batch:
                total_tokens += batch['labels'].ne(-100).sum().item()
            elif 'attention_mask' in batch:
                total_tokens += batch['attention_mask'].sum().item()
            else:
                total_tokens += batch['input_ids'].numel()

    denom = max(total_tokens if normalize_by_tokens else total_examples, 1)
    return hvp / float(denom)



def compute_max_eigenvalue(
    matrix_vector_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    neigs: int = 1,
    maxiter: int = 1000,
    tol: float = 1e-3,
    seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dominant eigenpair via power iteration on GPU.
      - neigs must be 1 (fast path for sharpness).
    """
    if neigs != 1:
        raise NotImplementedError("compute_max_eigenvalue supports neigs=1 only.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = torch.Generator(device=device).manual_seed(seed)

    v = torch.randn(dim, device=device, generator=g, dtype=torch.float32)
    v = v / (v.norm() + 1e-12)

    lam_old = None

    pbar = tqdm(total=maxiter, desc='Max eigen (power-iter)', mininterval=5.0)
    try:
        for _it in range(maxiter):
            # IMPORTANT: do NOT disable grads here; the HVP path needs autograd.
            w = matrix_vector_fn(v)               # should return a 1-D vector on the same device
            if w.dim() != 1:                      # be tolerant to shapes
                w = w.reshape(-1)

            # Rayleigh quotient and normalization; these ops don't need grads
            lam = (v @ w) / (v @ v + 1e-12)
            w_norm = w.norm() + 1e-12
            v = (w / w_norm)

            # iteration complete
            pbar.update(1)

            if lam_old is not None:
                rel = (lam - lam_old).abs() / (lam_old.abs() + 1e-6)
                if rel.item() < tol:
                    break
            lam_old = lam.detach()
    finally:
        # If we exited early, reset total so bar shows as full
        if pbar.n < pbar.total:
            pbar.total = pbar.n
            pbar.refresh()
        pbar.close()

    evals = lam_old.reshape(1)
    evecs = v.reshape(-1, 1).detach()
    return evals, evecs



def compute_trace_hutchplusplus(
    matrix_vector_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    num_samples: int
) -> float:
    """
    Compute trace of a matrix using Hutchinson++ algorithm.

    Args:
        matrix_vector_fn: Function that computes matrix-vector product (takes torch.Tensor, returns torch.Tensor)
        dim: Dimension of the matrix
        num_samples: Number of random samples to use

    Returns:
        Estimated trace as a float
    """
    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        hvp = matrix_vector_fn(gpu_vec)
        return hvp.detach().cpu().numpy()

    operator = LinearOperator((dim, dim), matvec=mv)
    tr = traceest(operator, num_samples)
    return float(tr)
