### Initial OTR Implementation

"""
chunked CE loss  +  One-Token-Roll-out (OTR) loss
"""

from typing import List, Optional
import torch
import torch.nn.functional as F

class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only up-casting one
    chunk at a time (from torchtune).
    """
    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
        use_dft: bool = False,
    ):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.use_dft = use_dft

    def compute_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="none"
        )
        if self.use_dft:
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1)
                valid_mask = labels != self.ignore_index
                label_probs = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                label_probs = label_probs * valid_mask
                label_probs = torch.where(
                    valid_mask, label_probs, torch.ones_like(label_probs)
                )
            ce_loss = ce_loss * label_probs
        return ce_loss.sum()

    def forward(
        self, logits: List[torch.Tensor], labels: torch.Tensor, reduction="sum"
    ) -> torch.Tensor:
        total_elements = (labels != self.ignore_index).sum()
        labels = [
            chunk.reshape(-1) for chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]
        logits = [
            chunk.reshape(-1, chunk.size(-1)) for chunk in logits
        ]
        total_loss = 0.0
        for logits_chunk, labels_chunk in zip(logits, labels):
            total_loss += self.compute_cross_entropy(logits_chunk, labels_chunk)
        if reduction == "sum":
            return total_loss
        return total_loss / total_elements

class OTRWithChunkedOutputLoss(torch.nn.Module):
    """
    One-Token-Roll-out loss with the same chunking strategy as CE.
    Eq. (8) / (9) of the paper, vectorised and memory-friendly.
    """
    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
        K: int = 8,          # # Monte-Carlo samples per token
        kappa: float = 1.2,  # temperature for exploration policy
        beta: float = -0.1,  # reward for non-GT tokens
    ):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.K = K
        self.kappa = kappa
        self.beta = beta

    # ---- core per-chunk routine ------------------------------------------------
    def compute_otr_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        logits : (N, |V|)   fp16/32
        labels : (N,)       ints     (may contain ignore_index)
        Returns
        -------
        scalar   – sum of token losses inside the chunk
        """
        device = logits.device
        valid_mask = labels.ne(self.ignore_index)          # (N,)
        if valid_mask.sum() == 0:
            return torch.zeros((), device=device)

        # Select only valid positions
        logits_valid  = logits[valid_mask].float()         # (M, V)
        labels_valid  = labels[valid_mask]                 # (M,)
        M, V = logits_valid.shape

        # Base log-probs  πθ  and exploration probs  πθ'   (Eq. 5)
        logp_base     = F.log_softmax(logits_valid,         dim=-1)    # (M, V)
        probs_explore = F.softmax(    logits_valid / self.kappa, dim=-1)    # (M, V)

        # Sample K candidate actions per token   (Eq. 6)
        actions = torch.multinomial(probs_explore, self.K, replacement=True)  # (M, K)

        # log πθ(a)  for those actions
        logp_actions = logp_base.gather(1, actions)                         # (M, K)

        # Rewards  R(a, x)  (Eq. 7)    – no grad through rewards
        gt = labels_valid.unsqueeze(1).expand_as(actions)
        rewards = torch.where(
            actions == gt,
            torch.ones_like(logp_actions),
            torch.full_like(logp_actions, self.beta),
        )

        # Token-level loss  –(1/K) Σ_j R log πθ(a_j)          (Eq. 8)
        loss_per_token = -(rewards.detach() * logp_actions).mean(dim=1)  # (M,)

        return loss_per_token.sum()     # sum over tokens

    # ---- outer chunk wrapper ---------------------------------------------------
    def forward(
        self, logits: List[torch.Tensor], labels: torch.Tensor, reduction="sum"
    ) -> torch.Tensor:
        """
        Same signature as CEWithChunkedOutputLoss so it can be swapped in place.
        """
        total_elements = (labels != self.ignore_index).sum()

        # Prepare chunked tensors exactly like in CE variant
        label_chunks = [
            chunk.reshape(-1) for chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]
        logit_chunks = [
            chunk.reshape(-1, chunk.size(-1)) for chunk in logits
        ]

        total_loss = 0.0
        for logits_chunk, labels_chunk in zip(logit_chunks, label_chunks):
            total_loss += self.compute_otr_loss(logits_chunk, labels_chunk)

        if reduction == "sum":
            return total_loss
        # mean over *tokens*, not over samples K (already averaged inside)
        return total_loss / total_elements


def _build_chunked_ce_loss_fn(
    num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
):
    loss_fn_ce = CEWithChunkedOutputLoss(num_output_chunks, ignore_index, use_dft)
    loss_fn_ce.compute_cross_entropy = torch.compile(
        loss_fn_ce.compute_cross_entropy, backend="inductor"
    )
    return loss_fn_ce


def _build_chunked_otr_loss_fn(
    num_output_chunks: int = 8,
    ignore_index: int = -100,
    K: int = 8,
    kappa: float = 1.2,
    beta: float = -0.1,
):
    loss_fn_otr = OTRWithChunkedOutputLoss(
        num_output_chunks=num_output_chunks,
        ignore_index=ignore_index,
        K=K,
        kappa=kappa,
        beta=beta,
    )
    loss_fn_otr.compute_otr_loss = torch.compile(
        loss_fn_otr.compute_otr_loss, backend="inductor"
    )
    return loss_fn_otr

def get_causal_lm_loss(
    num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
):
    """
    Original CE loss factory (kept for backward compatibility).
    """
    loss_fn_ce = _build_chunked_ce_loss_fn(num_output_chunks, ignore_index, use_dft)

    def chunked_fix_cross_entropy(
        source,
        target,
        num_items_in_batch: int = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        reduction = "sum" if num_items_in_batch is not None else "mean"
        logit_chunks = [chunk for chunk in source.chunk(loss_fn_ce.num_output_chunks, dim=1)]
        loss = loss_fn_ce(logit_chunks, target, reduction=reduction)
        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss

    def for_causal_lm_chunked_loss(
        logits,
        labels,
        vocab_size: int = None,
        num_items_in_batch: Optional[int] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if shift_labels is None:
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(logits.device)
        loss = chunked_fix_cross_entropy(
            logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
        )
        return loss

    return for_causal_lm_chunked_loss


def get_causal_lm_otr_loss(
    num_output_chunks: int = 8,
    ignore_index: int = -100,
    K: int = 8,
    kappa: float = 1.2,
    beta: float = -0.1,
):
    """
    Same as get_causal_lm_loss but uses the OTR objective internally.
    """
    loss_fn_otr = _build_chunked_otr_loss_fn(
        num_output_chunks, ignore_index, K, kappa, beta
    )

    def chunked_fix_otr(
        source,
        target,
        num_items_in_batch: int = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        reduction = "sum" if num_items_in_batch is not None else "mean"
        logit_chunks = [chunk for chunk in source.chunk(loss_fn_otr.num_output_chunks, dim=1)]
        loss = loss_fn_otr(logit_chunks, target, reduction=reduction)
        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss

    def for_causal_lm_chunked_otr_loss(
        logits,
        labels,
        vocab_size: int = None,
        num_items_in_batch: Optional[int] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if shift_labels is None:
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(logits.device)
        loss = chunked_fix_otr(
            logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
        )
        return loss

    return for_causal_lm_chunked_otr_loss

def patch_chunked_ce_loss_fn(
    num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
):
    """
    Keeps the original behaviour (CE).
    """
    import transformers.loss.loss_utils as _lu
    _lu.ForCausalLMLoss = get_causal_lm_loss(
        num_output_chunks, ignore_index, use_dft
    )
    _lu.LOSS_MAPPING["ForCausalLM"] = _lu.ForCausalLMLoss


def patch_chunked_otr_loss_fn(
    num_output_chunks: int = 8,
    ignore_index: int = -100,
    K: int = 8,
    kappa: float = 1.2,
    beta: float = -0.1,
):
    """
    Call this once *before* creating the Trainer to make HF/TRL use OTR.
    """
    import transformers.loss.loss_utils as _lu
    _lu.ForCausalLMLoss = get_causal_lm_otr_loss(
        num_output_chunks, ignore_index, K, kappa, beta
    )
    _lu.LOSS_MAPPING["ForCausalLM"] = _lu.ForCausalLMLoss


### Original Axolotl Chunked Loss

# """
# chunked ce loss
# """

# from typing import List, Optional

# import torch
# import torch.nn.functional as F


# # copied and modified from torchtune.modules.loss.CEWithChunkedOutputLoss
# class CEWithChunkedOutputLoss(torch.nn.Module):
#     """
#     Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

#     For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
#     """

#     def __init__(
#         self,
#         num_output_chunks: int = 8,
#         ignore_index: int = -100,
#         use_dft: bool = False,
#     ):
#         super().__init__()
#         self.num_output_chunks = num_output_chunks
#         self.ignore_index = ignore_index
#         self.use_dft = use_dft

#     def compute_cross_entropy(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         normalize: bool = True,
#     ) -> torch.Tensor:
#         """
#         Upcast logits to fp32 and compute cross entropy loss.
#         """
#         ce_loss = F.cross_entropy(
#             logits.float(), labels, ignore_index=self.ignore_index, reduction="none"
#         )

#         if self.use_dft:
#             # Compute probabilities and gather the ones corresponding to labels
#             with torch.no_grad():  # Stop gradient
#                 probs = torch.softmax(logits.float(), dim=-1)
#                 # Create mask for valid tokens (not ignore_index)
#                 valid_mask = labels != self.ignore_index
#                 # Gather probabilities for the correct tokens
#                 label_probs = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
#                 # Apply mask to only scale valid tokens
#                 label_probs = label_probs * valid_mask
#                 # Avoid multiplication by 0 for ignored tokens
#                 label_probs = torch.where(
#                     valid_mask, label_probs, torch.ones_like(label_probs)
#                 )

#             # Scale the loss by the probability (DFT)
#             ce_loss = ce_loss * label_probs

#         return ce_loss.sum()

#     def forward(
#         self, logits: List[torch.Tensor], labels: torch.Tensor, reduction="sum"
#     ) -> torch.Tensor:
#         """
#         Args:
#             logits (List[torch.Tensor]): List of chunked logits of length
#                 ``self.num_output_chunks``, where each chunk has shape
#                 ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
#             labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.
#             reduction (str): The reduction to apply to the output.

#         Returns:
#             torch.Tensor: Cross entropy loss of shape (1,).
#         """

#         total_elements = (labels != self.ignore_index).sum()

#         # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
#         labels = [
#             target_chunk.reshape(-1)
#             for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
#         ]
#         # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
#         logits = [
#             logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
#         ]

#         # compute one chunk at a time
#         total_loss = 0.0
#         for logits_chunk, labels_chunk in zip(logits, labels, strict=False):
#             total_loss += self.compute_cross_entropy(logits_chunk, labels_chunk)

#         if reduction == "sum":
#             return total_loss
#         return total_loss / total_elements


# def _build_chunked_ce_loss_fn(
#     num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
# ):
#     loss_fn_ce = CEWithChunkedOutputLoss(num_output_chunks, ignore_index, use_dft)
#     loss_fn_ce.compute_cross_entropy = torch.compile(
#         loss_fn_ce.compute_cross_entropy, backend="inductor"
#     )
#     return loss_fn_ce


# def get_causal_lm_loss(
#     num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
# ):
#     loss_fn_ce = _build_chunked_ce_loss_fn(num_output_chunks, ignore_index, use_dft)

#     def chunked_fix_cross_entropy(
#         source,
#         target,
#         num_items_in_batch: int = None,
#         ignore_index: int = -100,
#         **kwargs,
#     ):
#         reduction = "sum" if num_items_in_batch is not None else "mean"
#         logit_chunks = [
#             chunk for chunk in source.chunk(loss_fn_ce.num_output_chunks, dim=1)
#         ]
#         loss = loss_fn_ce(logit_chunks, target, reduction=reduction)
#         if reduction == "sum":
#             loss = loss / num_items_in_batch
#         return loss

#     def for_causal_lm_chunked_loss(
#         logits,
#         labels,
#         vocab_size: int = None,
#         num_items_in_batch: Optional[int] = None,
#         ignore_index: int = -100,
#         shift_labels: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> torch.Tensor:
#         # skip the upcast to float since we handle that in the chunking loss
#         if shift_labels is None:
#             # Shift so that tokens < n predict n
#             labels = F.pad(labels, (0, 1), value=ignore_index)
#             shift_labels = labels[..., 1:].contiguous()

#         # Skip Flattening the tokens
#         # Enable model parallelism
#         shift_labels = shift_labels.to(logits.device)
#         loss = chunked_fix_cross_entropy(
#             logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
#         )
#         return loss

#     return for_causal_lm_chunked_loss


# def patch_chunked_ce_loss_fn(
#     num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
# ):
#     import transformers.loss.loss_utils

#     for_causal_lm_chunked_loss = get_causal_lm_loss(
#         num_output_chunks, ignore_index, use_dft
#     )
#     transformers.loss.loss_utils.ForCausalLMLoss = for_causal_lm_chunked_loss
#     transformers.loss.loss_utils.LOSS_MAPPING["ForCausalLM"] = (
#         for_causal_lm_chunked_loss
#     )
