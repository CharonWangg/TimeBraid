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

"""Abstract configs for TimesFM layers."""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class ResidualBlockConfig:
  """Framework-agnostic config for a residual block."""

  input_dims: int
  hidden_dims: int
  output_dims: int
  use_bias: bool
  activation: Literal["relu", "swish", "none"]


@dataclasses.dataclass(frozen=True)
class RandomFourierFeaturesConfig:
  """Framework-agnostic config for random fourier features."""

  input_dims: int
  output_dims: int
  projection_stddev: float
  use_bias: bool


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
  """Framework-agnostic config for a transformer."""

  model_dims: int
  hidden_dims: int
  num_heads: int
  attention_norm: Literal["rms"]
  feedforward_norm: Literal["rms"]
  qk_norm: Literal["rms", "none"]
  use_bias: bool
  use_rotary_position_embeddings: bool
  ff_activation: Literal["relu", "swish", "none"]
  fuse_qkv: bool


@dataclasses.dataclass(frozen=True)
class StackedTransformersConfig:
  """Framework-agnostic config for a stacked transformers."""

  num_layers: int
  transformer: TransformerConfig
