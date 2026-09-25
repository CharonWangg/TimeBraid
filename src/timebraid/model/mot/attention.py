"""Mixed language and time-series attention for TimeBraid."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _GlobalCausalJointSelfAttention(nn.Module):
    """Projection-free packed-attention kernel shared by native and residual paths."""

    def __init__(
        self,
        q_heads: int,
        q_head_dim: int,
        *,
        lang_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.q_heads = q_heads
        self.q_head_dim = q_head_dim
        self.q_attention_hidden_size = self.q_heads * self.q_head_dim
        # Residual fusion projects language, generation-TS and understanding-TS
        # into one shared Qwen head geometry before packing, so its softmax scale
        # is tied to `q_head_dim`. Native language attention keeps Qwen's
        # `self_attn.scaling`; native TimesFM replay uses its own head dimension
        # for the fixed attention-kernel scale.
        self.scale = float(self.q_head_dim) ** -0.5
        self.lang_hidden_size = int(lang_hidden_size or self.q_attention_hidden_size)
        if self.lang_hidden_size <= 0:
            raise ValueError(
                f"`lang_hidden_size` must be positive, got {self.lang_hidden_size}."
            )
        self._flash_attn_varlen_func = None
        self._fa_peft_integration_check = None

    def _get_flash_attn_varlen_func(self):
        if self._flash_attn_varlen_func is not None:
            return self._flash_attn_varlen_func
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception as exc:
            raise RuntimeError(
                "MoT mixed attention is FA2-only but `flash_attn` could not be imported. "
                "Install flash-attn in the active TimeBraid Python environment."
            ) from exc
        self._flash_attn_varlen_func = flash_attn_varlen_func
        return self._flash_attn_varlen_func

    def _get_fa_peft_integration_check(self):
        if self._fa_peft_integration_check is not None:
            return self._fa_peft_integration_check
        try:
            from transformers.modeling_flash_attention_utils import (
                fa_peft_integration_check,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to import transformers.modeling_flash_attention_utils.fa_peft_integration_check. "
                "This runtime patch follows HF FA2 dtype reconciliation logic and requires this symbol."
            ) from exc
        self._fa_peft_integration_check = fa_peft_integration_check
        return self._fa_peft_integration_check

    def _infer_fa2_target_dtype_like_hf(self, q: torch.Tensor) -> Optional[torch.dtype]:
        """Mirror HF FA2 target-dtype inference for packed-attention calls."""
        target_dtype: Optional[torch.dtype] = None
        if q.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self, "config") and hasattr(
                self.config, "_pre_quantization_dtype"
            ):
                target_dtype = self.config._pre_quantization_dtype
            else:
                first_linear = next(
                    (layer for layer in self.modules() if isinstance(layer, nn.Linear)),
                    None,
                )
                if first_linear is None:
                    raise RuntimeError(
                        "Cannot infer FA2 target dtype because no nn.Linear module was found in bridge layer."
                    )
                target_dtype = first_linear.weight.dtype
        return target_dtype

    def _reconcile_fa2_qkv(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mode_label: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = self._infer_fa2_target_dtype_like_hf(q)
        fa_peft_integration_check = self._get_fa_peft_integration_check()
        q_fa, k_fa, v_fa = fa_peft_integration_check(q, k, v, target_dtype)
        if q_fa.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                f"MoT mixed FA2 {mode_label} requires fp16/bf16 Q/K/V after HF-style dtype reconciliation, "
                f"but got dtype={q_fa.dtype}, original_dtype={q.dtype}, "
                f"target_dtype={target_dtype}, autocast_enabled={torch.is_autocast_enabled()}."
            )
        return q_fa, k_fa, v_fa

    def _reconcile_fast_qkv(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mode_label: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize Q/K/V for the shared FA2 packed-attention kernel."""
        return self._reconcile_fa2_qkv(q=q, k=k, v=v, mode_label=mode_label)

    def _cu_seqlens(self, *, seqlens: list[int], device: torch.device) -> torch.Tensor:
        cu = [0]
        for seq_len in seqlens:
            cu.append(cu[-1] + int(seq_len))
        return torch.tensor(cu, device=device, dtype=torch.int32)

    def _seqlens_from_cu_seqlens(self, *, cu_seqlens: torch.Tensor) -> list[int]:
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] == 0:
            raise RuntimeError(
                "Expected a non-empty 1D cu_seqlens tensor, got "
                f"shape={tuple(cu_seqlens.shape)}."
            )
        cu_values = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
        return [
            int(end) - int(start) for start, end in zip(cu_values[:-1], cu_values[1:])
        ]

    def _run_varlen_flash_attention(
        self,
        *,
        q_sorted: torch.Tensor,
        k_sorted: torch.Tensor,
        v_sorted: torch.Tensor,
        q_positions_sorted: torch.Tensor,
        k_positions_sorted: torch.Tensor,
        softmax_scale: float,
        dropout_p: float,
        mode_label: str,
        q_seqlens: Optional[list[int]] = None,
        k_seqlens: Optional[list[int]] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        if q_sorted.ndim != 3 or k_sorted.ndim != 3 or v_sorted.ndim != 3:
            raise RuntimeError(
                "Fast MoT attention expects packed [N,H,D] Q/K/V tensors, got "
                f"q={tuple(q_sorted.shape)}, k={tuple(k_sorted.shape)}, v={tuple(v_sorted.shape)}."
            )
        if k_sorted.shape != v_sorted.shape:
            raise RuntimeError(
                "Fast MoT attention expects aligned K/V shapes, got "
                f"k={tuple(k_sorted.shape)}, v={tuple(v_sorted.shape)}."
            )
        if (
            q_positions_sorted.ndim != 1
            or q_positions_sorted.shape[0] != q_sorted.shape[0]
        ):
            raise RuntimeError(
                "Fast MoT attention expects 1D query positions aligned to Q, got "
                f"positions={tuple(q_positions_sorted.shape)}, q_tokens={q_sorted.shape[0]}."
            )
        if (
            k_positions_sorted.ndim != 1
            or k_positions_sorted.shape[0] != k_sorted.shape[0]
        ):
            raise RuntimeError(
                "Fast MoT attention expects 1D key positions aligned to K, got "
                f"positions={tuple(k_positions_sorted.shape)}, k_tokens={k_sorted.shape[0]}."
            )

        if q_seqlens is None:
            if cu_seqlens_q is None:
                raise RuntimeError(
                    "Fast MoT attention requires either q_seqlens or cu_seqlens_q."
                )
            q_seqlens = self._seqlens_from_cu_seqlens(cu_seqlens=cu_seqlens_q)
        if k_seqlens is None:
            if cu_seqlens_k is None:
                raise RuntimeError(
                    "Fast MoT attention requires either k_seqlens or cu_seqlens_k."
                )
            k_seqlens = self._seqlens_from_cu_seqlens(cu_seqlens=cu_seqlens_k)
        if max_seqlen_q is None:
            max_seqlen_q = max(q_seqlens) if q_seqlens else 0
        if max_seqlen_k is None:
            max_seqlen_k = max(k_seqlens) if k_seqlens else 0

        if q_sorted.device.type != "cuda":
            raise RuntimeError(
                f"MoT FA2 {mode_label} requires CUDA device, got {q_sorted.device.type}."
            )
        q_fast, k_fast, v_fast = self._reconcile_fast_qkv(
            q=q_sorted,
            k=k_sorted,
            v=v_sorted,
            mode_label=mode_label,
        )
        if cu_seqlens_q is None:
            cu_seqlens_q = self._cu_seqlens(seqlens=q_seqlens, device=q_sorted.device)
        if cu_seqlens_k is None:
            cu_seqlens_k = self._cu_seqlens(seqlens=k_seqlens, device=q_sorted.device)
        return self._get_flash_attn_varlen_func()(
            q_fast,
            k_fast,
            v_fast,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=True,
        )

    def run_packed_self_attention(
        self,
        *,
        q_sorted: torch.Tensor,
        k_sorted: torch.Tensor,
        v_sorted: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_positions_sorted: torch.Tensor,
        softmax_scale: float,
        dropout_p: float,
        mode_label: str,
    ) -> torch.Tensor:
        """Shared FA2 packed-attention entrypoint for native and residual callers."""
        return self._run_varlen_flash_attention(
            q_sorted=q_sorted,
            k_sorted=k_sorted,
            v_sorted=v_sorted,
            q_positions_sorted=rope_positions_sorted,
            k_positions_sorted=rope_positions_sorted,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            mode_label=mode_label,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )


class ResidualStreamProjection(nn.Module):
    """Project one stream into and out of the shared residual-attention heads."""

    def __init__(
        self,
        *,
        input_size: int,
        attention_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")
        if attention_heads <= 0:
            raise ValueError(
                f"attention_heads must be positive, got {attention_heads}."
            )
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")

        self.input_size = int(input_size)
        self.attention_heads = int(attention_heads)
        self.head_dim = int(head_dim)
        attention_size = self.attention_heads * self.head_dim

        # Keep registration order identical across all streams. Q/K use the
        # Qwen-style post-projection head norm; V remains the raw projection.
        #
        # The epsilon is fixed at 1e-6 on purpose rather than mirrored from the
        # host LM's `rms_norm_eps`. These norms carry trained weights, so the
        # value they were trained under is part of this module's topology, the
        # same as its head dimension; reading it from whatever backbone happens
        # to be loaded would change a trained module's numerics. It matches
        # Qwen3's default, so the two agree in practice today. Persisting it in
        # the MoT restore contract is the proper fix and is deferred to the
        # change that can update the training tree and the published configs
        # together.
        self.input_norm = nn.RMSNorm(
            self.input_size, eps=1.0e-6, elementwise_affine=True
        )
        self.q_proj = nn.Linear(self.input_size, attention_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1.0e-6, elementwise_affine=True)
        self.k_proj = nn.Linear(self.input_size, attention_size, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1.0e-6, elementwise_affine=True)
        self.v_proj = nn.Linear(self.input_size, attention_size, bias=False)
        self.out_proj = nn.Linear(attention_size, self.input_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    @staticmethod
    def _apply_head_norm(states: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        normalized = norm(states.to(dtype=norm.weight.dtype))
        return normalized.to(dtype=states.dtype)

    def project_qkv(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2 or int(hidden.shape[-1]) != self.input_size:
            raise RuntimeError(
                "Residual stream projection expects [N,H], got "
                f"shape={tuple(hidden.shape)}, expected_hidden={self.input_size}."
            )
        if int(hidden.shape[0]) == 0:
            empty = hidden.new_empty((0, self.attention_heads, self.head_dim))
            return empty, empty, empty

        normalized = self.input_norm(hidden.to(dtype=self.input_norm.weight.dtype)).to(
            dtype=self.q_proj.weight.dtype
        )
        token_count = int(normalized.shape[0])
        head_shape = (token_count, self.attention_heads, self.head_dim)
        q = self.q_proj(normalized).view(head_shape)
        k = self.k_proj(normalized).view(head_shape)
        v = self.v_proj(normalized).view(head_shape)
        q = self._apply_head_norm(q, self.q_norm)
        k = self._apply_head_norm(k, self.k_norm)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def project_output(
        self, out_heads: torch.Tensor, *, output_dtype: torch.dtype
    ) -> torch.Tensor:
        expected_head_shape = (self.attention_heads, self.head_dim)
        if out_heads.ndim != 3 or tuple(out_heads.shape[1:]) != expected_head_shape:
            raise RuntimeError(
                "Residual stream output expects [N,H,D], got "
                f"shape={tuple(out_heads.shape)}, expected_heads={expected_head_shape}."
            )
        flat = out_heads.reshape(out_heads.shape[0], -1)
        delta = self.out_proj(flat.to(dtype=self.out_proj.weight.dtype))
        return delta.to(dtype=output_dtype)


class GlobalResidualAttention(_GlobalCausalJointSelfAttention):
    """One joint residual-attention block over language and paired TS streams."""

    def __init__(
        self,
        *,
        language_hidden_size: int,
        generation_hidden_size: Optional[int],
        understanding_hidden_size: Optional[int],
        attention_heads: int,
        head_dim: int,
    ) -> None:
        if generation_hidden_size is None and understanding_hidden_size is None:
            raise ValueError(
                "GlobalResidualAttention requires generation or understanding."
            )
        super().__init__(
            q_heads=attention_heads,
            q_head_dim=head_dim,
            lang_hidden_size=language_hidden_size,
        )

        # Module registration order is part of the checkpoint schema.
        self.language = ResidualStreamProjection(
            input_size=language_hidden_size,
            attention_heads=attention_heads,
            head_dim=head_dim,
        )
        self.generation = (
            ResidualStreamProjection(
                input_size=generation_hidden_size,
                attention_heads=attention_heads,
                head_dim=head_dim,
            )
            if generation_hidden_size is not None
            else None
        )
        self.understanding = (
            ResidualStreamProjection(
                input_size=understanding_hidden_size,
                attention_heads=attention_heads,
                head_dim=head_dim,
            )
            if understanding_hidden_size is not None
            else None
        )
        self.residual_attention_mode = True

    @staticmethod
    def _require_stream(
        stream: Optional[ResidualStreamProjection], *, name: str
    ) -> ResidualStreamProjection:
        if stream is None:
            raise RuntimeError(
                f"GlobalResidualAttention has no {name} stream on this layer."
            )
        return stream

    def project_language_qkv(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.language.project_qkv(hidden)

    def project_generation_qkv(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._require_stream(self.generation, name="generation").project_qkv(
            hidden
        )

    def project_understanding_qkv(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._require_stream(
            self.understanding, name="understanding"
        ).project_qkv(hidden)

    def project_language_delta(
        self, out_heads: torch.Tensor, *, output_dtype: torch.dtype
    ) -> torch.Tensor:
        return self.language.project_output(out_heads, output_dtype=output_dtype)

    def project_generation_delta(
        self, out_heads: torch.Tensor, *, output_dtype: torch.dtype
    ) -> torch.Tensor:
        return self._require_stream(self.generation, name="generation").project_output(
            out_heads, output_dtype=output_dtype
        )

    def project_understanding_delta(
        self, out_heads: torch.Tensor, *, output_dtype: torch.dtype
    ) -> torch.Tensor:
        return self._require_stream(
            self.understanding, name="understanding"
        ).project_output(out_heads, output_dtype=output_dtype)


__all__ = [
    "GlobalResidualAttention",
    "ResidualStreamProjection",
    "_GlobalCausalJointSelfAttention",
]
