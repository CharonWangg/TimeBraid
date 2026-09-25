import pytest
import torch

from timebraid.model.mot.attention import (
    GlobalResidualAttention,
    ResidualStreamProjection,
)

_LEAF_NAMES = (
    "input_norm",
    "q_proj",
    "q_norm",
    "k_proj",
    "k_norm",
    "v_proj",
    "out_proj",
)


def _make_global_residual_attention(
    *, generation: bool = True, understanding: bool = True
) -> GlobalResidualAttention:
    return GlobalResidualAttention(
        language_hidden_size=8,
        generation_hidden_size=3 if generation else None,
        understanding_hidden_size=5 if understanding else None,
        attention_heads=2,
        head_dim=4,
    )


def test_global_residual_attention_has_one_explicit_module_per_stream():
    attention = _make_global_residual_attention()

    assert isinstance(attention.language, ResidualStreamProjection)
    assert isinstance(attention.generation, ResidualStreamProjection)
    assert isinstance(attention.understanding, ResidualStreamProjection)
    assert list(attention._modules) == [
        "language",
        "generation",
        "understanding",
    ]
    for stream_name in ("language", "generation", "understanding"):
        stream = getattr(attention, stream_name)
        assert list(stream._modules) == list(_LEAF_NAMES)

    state_keys = list(attention.state_dict())
    assert state_keys == [
        f"{stream_name}.{leaf_name}.weight"
        for stream_name in ("language", "generation", "understanding")
        for leaf_name in _LEAF_NAMES
    ]
    assert not any("residual_attn_" in key for key in state_keys)


@pytest.mark.parametrize(
    ("generation", "understanding"),
    [(True, False), (False, True), (True, True)],
)
def test_global_residual_attention_registers_only_paired_ts_streams(
    generation, understanding
):
    attention = _make_global_residual_attention(
        generation=generation,
        understanding=understanding,
    )

    assert (attention.generation is not None) is generation
    assert (attention.understanding is not None) is understanding
    assert any(key.startswith("language.") for key in attention.state_dict())
    assert (
        any(key.startswith("generation.") for key in attention.state_dict())
        is generation
    )
    assert (
        any(key.startswith("understanding.") for key in attention.state_dict())
        is understanding
    )


def test_global_residual_attention_requires_at_least_one_ts_stream():
    with pytest.raises(
        ValueError,
        match="requires generation or understanding",
    ):
        _make_global_residual_attention(generation=False, understanding=False)


def test_residual_stream_projection_normalizes_qk_but_not_v():
    torch.manual_seed(7)
    projection = ResidualStreamProjection(
        input_size=6,
        attention_heads=2,
        head_dim=3,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        projection.q_norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5]))
        projection.k_norm.weight.copy_(torch.tensor([1.5, 1.0, 0.5]))

    hidden = torch.randn(4, 6, dtype=torch.float32)
    normalized = projection.input_norm(hidden.to(dtype=torch.float64))
    raw_q = projection.q_proj(normalized).view(4, 2, 3)
    raw_k = projection.k_proj(normalized).view(4, 2, 3)
    raw_v = projection.v_proj(normalized).view(4, 2, 3)

    q, k, v = projection.project_qkv(hidden)

    assert q.dtype == k.dtype == v.dtype == torch.float64
    torch.testing.assert_close(q, projection.q_norm(raw_q))
    torch.testing.assert_close(k, projection.k_norm(raw_k))
    torch.testing.assert_close(v, raw_v)


def test_residual_stream_projection_zero_initializes_only_output_projection():
    torch.manual_seed(11)
    projection = ResidualStreamProjection(
        input_size=6,
        attention_heads=2,
        head_dim=3,
    )

    assert torch.count_nonzero(projection.out_proj.weight).item() == 0
    for name in ("q_proj", "k_proj", "v_proj"):
        assert torch.count_nonzero(getattr(projection, name).weight).item() > 0

    heads = torch.randn(4, 2, 3, dtype=torch.float64)
    delta = projection.project_output(heads, output_dtype=torch.float64)
    assert delta.dtype == torch.float64
    assert torch.count_nonzero(delta).item() == 0


def test_absent_stream_fails_instead_of_falling_back_to_another_projection():
    attention = _make_global_residual_attention(
        generation=True,
        understanding=False,
    )

    with pytest.raises(RuntimeError, match="no understanding stream"):
        attention.project_understanding_qkv(torch.randn(2, 5))
