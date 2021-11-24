# Copyright 2021 The Trieste Contributors
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

from __future__ import annotations

from typing import Tuple, Type

import gpflow
import pytest
from gpflow.models import GPR, SGPR, SVGP, VGP

from trieste.models import TrainableProbabilisticModel
from trieste.models.config import ModelRegistry
from trieste.models.gpflow import (
    GaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.optimizer import BatchOptimizer, Optimizer


@pytest.mark.parametrize(
    "supported_models",
    [
        (GPR, GaussianProcessRegression, Optimizer),
        (SGPR, GaussianProcessRegression, Optimizer),
        (VGP, VariationalGaussianProcess, Optimizer),
        (SVGP, SparseVariational, BatchOptimizer),
    ],
)
def test_supported_gpflow_models_are_correctly_registered(
    supported_models: Tuple[
        Type[gpflow.models.GPModel], Type[TrainableProbabilisticModel], Type[Optimizer]
    ]
) -> None:

    model_type, interface, optimizer = supported_models

    assert model_type in ModelRegistry.get_registered_models()
    assert ModelRegistry.get_interface(model_type) == interface
    assert ModelRegistry.get_optimizer(model_type) == optimizer
