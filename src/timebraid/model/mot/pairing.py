"""Layer-pairing helpers for the TimeBraid model core."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_QWEN_PAIRABLE_LAYER_TYPE = "full_attention"


def _build_pair_map(
    num_q_layers: int,
    num_t_layers: int,
    pairing_mode: str,
    q_layer_indices: Optional[List[int]] = None,
) -> Dict[int, int]:
    """Build the Q-layer -> TimesFM-layer mapping for one pairing mode."""
    if num_t_layers <= 0:
        raise ValueError(f"num_t_layers must be positive, got {num_t_layers}")
    if q_layer_indices is None:
        available_q_layers = list(range(num_q_layers))
    else:
        available_q_layers = [int(idx) for idx in q_layer_indices]
    if not available_q_layers:
        raise ValueError("No pairable Q layers are available for MoT bridge mapping.")
    if available_q_layers != sorted(available_q_layers):
        raise ValueError(
            f"q_layer_indices must be sorted ascending, got {available_q_layers}"
        )
    if available_q_layers[0] < 0 or available_q_layers[-1] >= num_q_layers:
        raise ValueError(
            "q_layer_indices must stay inside [0, num_q_layers). "
            f"got indices={available_q_layers}, q_layers={num_q_layers}"
        )

    if len(available_q_layers) > len(set(available_q_layers)):
        raise ValueError(f"q_layer_indices must be unique, got {available_q_layers}")
    if num_t_layers > num_q_layers and q_layer_indices is None:
        raise ValueError(
            f"TimesFM layer count exceeds Qwen layers: t_layers={num_t_layers}, q_layers={num_q_layers}"
        )

    if pairing_mode == "interleaved":
        if len(available_q_layers) < num_t_layers:
            raise ValueError(
                "`interleaved` requires at least as many pairable Qwen layers as active TimesFM layers, "
                f"got pairable_q_layers={len(available_q_layers)}, t_layers={num_t_layers}."
            )
        if num_t_layers == 1:
            return {available_q_layers[0]: 0}
        q_last = len(available_q_layers) - 1
        t_last = num_t_layers - 1
        selected_q_layers = [
            available_q_layers[round(t_idx * q_last / t_last)]
            for t_idx in range(num_t_layers)
        ]
        return {q_idx: t_idx for t_idx, q_idx in enumerate(selected_q_layers)}

    if pairing_mode == "dsfp":
        if len(available_q_layers) != num_q_layers:
            raise ValueError(
                "`dsfp` requires every Qwen layer to be pairable, got "
                f"pairable_q_layers={len(available_q_layers)}, q_layers={num_q_layers}."
            )
        if num_t_layers != num_q_layers:
            raise ValueError(
                "`dsfp` maps the expanded TimesFM stack one-to-one onto Qwen depth, got "
                f"t_layers={num_t_layers}, q_layers={num_q_layers}."
            )
        return {q_idx: t_idx for t_idx, q_idx in enumerate(available_q_layers)}

    raise ValueError(
        f"Unsupported MoT pairing mode: {pairing_mode!r}. Expected 'interleaved' or 'dsfp'."
    )


def _resolve_pairable_q_layer_indices(config: Any, num_q_layers: int) -> List[int]:
    """Resolve which Qwen decoder layers can participate in the TimeBraid layer plan."""
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return list(range(num_q_layers))
    if len(layer_types) != num_q_layers:
        raise ValueError(
            "config.layer_types length mismatch with num_hidden_layers: "
            f"len(layer_types)={len(layer_types)}, num_hidden_layers={num_q_layers}"
        )

    pairable = [
        idx
        for idx, layer_type in enumerate(layer_types)
        if layer_type == _QWEN_PAIRABLE_LAYER_TYPE
    ]
    if not pairable:
        raise ValueError(
            f"MoT requires at least one `{_QWEN_PAIRABLE_LAYER_TYPE}` layer, but none were found in "
            "config.layer_types."
        )
    return pairable
