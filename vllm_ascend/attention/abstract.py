# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch


class DSAAttentionImpl(ABC):

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...
