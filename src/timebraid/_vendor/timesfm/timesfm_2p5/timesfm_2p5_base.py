# Modified by TimeBraid contributors for the vendored PyTorch-only runtime.
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimesFM 2p5 base implementation."""

import dataclasses

from .. import configs

ResidualBlockConfig = configs.ResidualBlockConfig
StackedTransformersConfig = configs.StackedTransformersConfig
TransformerConfig = configs.TransformerConfig


@dataclasses.dataclass(frozen=True)
class TimesFM_2p5_200M_Definition:
  """Framework-agnostic config of TimesFM 2.5."""

  context_limit = 16384
  input_patch_len: int = 32
  output_patch_len: int = 128
  output_quantile_len: int = 1024
  quantiles: list[float] = dataclasses.field(
    default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  )
  decode_index: int = 5
  tokenizer: ResidualBlockConfig = ResidualBlockConfig(
    input_dims=64,
    hidden_dims=1280,
    output_dims=1280,
    use_bias=True,
    activation="swish",
  )
  stacked_transformers: StackedTransformersConfig = StackedTransformersConfig(
    num_layers=20,
    transformer=TransformerConfig(
      model_dims=1280,
      hidden_dims=1280,
      num_heads=16,
      attention_norm="rms",
      feedforward_norm="rms",
      qk_norm="rms",
      use_bias=False,
      use_rotary_position_embeddings=True,
      ff_activation="swish",
      fuse_qkv=True,
    ),
  )
  output_projection_point: ResidualBlockConfig = ResidualBlockConfig(
    input_dims=1280,
    hidden_dims=1280,
    output_dims=1280,
    use_bias=False,
    activation="swish",
  )
  output_projection_quantiles: ResidualBlockConfig = ResidualBlockConfig(
    input_dims=1280,
    hidden_dims=1280,
    output_dims=10240,
    use_bias=False,
    activation="swish",
  )
