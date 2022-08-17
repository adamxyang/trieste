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
"""
This module contains acquisition function builders, which build and define our acquisition
functions --- functions that estimate the utility of evaluating sets of candidate points.
"""
from __future__ import annotations

from typing import Mapping, Optional, cast

import tensorflow as tf
import tensorflow_probability as tfp

from ..optimizer import generate_random_search_optimizer, batchify_vectorize

from ...data import Dataset
from ...models import ProbabilisticModel, ReparametrizationSampler
from ...models.interfaces import (
    HasReparamSampler,
    SupportsGetObservationNoise,
    SupportsReparamSamplerObservationNoise,
)
from ...space import SearchSpace
from ...types import TensorType
from ...utils import DEFAULTS
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
    ProbabilisticModelType,
    SingleModelAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
    UpdatablePenalizationFunction
)
from .greedy_batch import LocalPenalization, local_penalizer, soft_local_penalizer
import trieste
from .function import ExpectedImprovement, MonteCarloExpectedImprovement

import copy

from trieste.acquisition.rule import EfficientGlobalOptimization



class Glasses(SingleModelAcquisitionBuilder[ProbabilisticModel]):

    def __init__(
        self,
        search_space: SearchSpace,
        num_lookahead: int
        ):
        self._search_space = search_space
        self._num_lookahead = num_lookahead

    def __repr__(self) -> str:
        """"""
        return "Glasses()"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return glasses(model, eta, self._search_space, self._num_lookahead, dataset)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer.  Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, glasses), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        function.update(eta, dataset)  # type: ignore
        return function




class glasses(AcquisitionFunctionClass):
    def __init__(
        self, 
        model: ProbabilisticModel, 
        eta: TensorType, 
        search_space: SearchSpace, 
        num_lookahead: int,
        dataset: Dataset,
        ):
        r"""
        :param model: The model of the objective function.
        :param eta: The "best" observation.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        self._model = model
        self._eta = tf.Variable(eta)
        self._search_space = search_space
        self._num_lookahead = num_lookahead
        self._dataset = dataset
        # self._optimizer = generate_random_search_optimizer(100) # 100 samples (make user custom)
        self._optimizer = batchify_vectorize(generate_random_search_optimizer(100), batch_size=100) # 100 samples (make user custom)
        self._inner_acq_builder = ExpectedImprovement()
        self._inner_acq = self._inner_acq_builder.prepare_acquisition_function(self._model, self._dataset)
        self._jitter = DEFAULTS.JITTER

        samples = self._search_space.sample(num_samples=100)
        lipschitz, eta = self._get_lipschitz_estimate(self._model, samples)
        self._lipschitz = tf.Variable(lipschitz)

    def update(self, eta: TensorType, dataset: Dataset) -> None:
        """Update the acquisition function with a new eta value."""
        self._eta.assign(eta)
        self._dataset = dataset
        # update is called for new BO steps
        samples = self._search_space.sample(num_samples=100)
        lipschitz, eta = self._get_lipschitz_estimate(self._model, samples)
        self._lipschitz.assign(lipschitz)
        self._inner_acq = self._inner_acq_builder.update_acquisition_function(self._inner_acq, self._model, self._dataset)

    def _get_lipschitz_estimate(self, model, sampled_points):
        with tf.GradientTape() as g:
            g.watch(sampled_points)
            mean, _ = model.predict(sampled_points)
        grads = g.gradient(mean, sampled_points)
        grads_norm = tf.norm(grads, axis=1)
        max_grads_norm = tf.reduce_max(grads_norm)
        eta = tf.reduce_min(mean, axis=0)
        return max_grads_norm, eta

        

    #@tf.function
    def __call__(self, x: TensorType) -> TensorType: # [N, 1, d] -> [N, 1]
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        pending_points = x
        all_points = x
        # local_penalty = soft_local_penalizer(self._model, pending_points, self._lipschitz, self._eta)

        class acq_with_penalty:
            def __init__(self, pending_points, step, model=self._model, lipschitz=self._lipschitz, eta=self._eta, inner_acq=self._inner_acq):
                self._pending_points = pending_points
                self._step = step
                self._acq = 1
                self._model = model
                self._lipschitz = lipschitz
                self._eta = eta
                self._inner_acq = inner_acq
                
            def __call__(self, x):   
                mean_pending, variance_pending = self._model.predict(self._pending_points)
                radius = tf.Variable(
                    tf.transpose((mean_pending - self._eta) / self._lipschitz),
                )
                scale = tf.Variable(
                    tf.transpose(tf.sqrt(variance_pending) / self._lipschitz),
                )
                pairwise_distances = tf.norm(tf.squeeze(x)[:,None,:] - tf.squeeze(self._pending_points)[None,:,:], axis=-1)
                standardised_distances = (tf.squeeze(pairwise_distances) - tf.squeeze(radius)[None,:]) / tf.squeeze(scale)[None,:]
                normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
                penalization = normal.cdf(standardised_distances)
                if step == 0:
                    mean, variance = self._model.predict(tf.squeeze(x))
                    normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
                    EI = (self._eta - mean) * normal.cdf(self._eta) + variance * normal.prob(self._eta)
                    self._acq = tf.squeeze(EI)[:,None,None] * penalization[:,:,None]
                    return self._acq
                else:
                    return self._acq * penalization[:,:,None]   # BxNx1, N outside, B inside

        sampler = self._model.reparam_sampler(200)   # MC samples to evaluate EI, make user custom
        all_obs = []

        x_in = self._search_space.sample(num_samples=1000)
        for step in range(self._num_lookahead):
            # pending_points = self._optimizer(self._search_space, acq_with_penalty(pending_points, step))[None,:,:]

            obs_at_query_points = tf.squeeze(sampler.sample(tf.squeeze(pending_points), jitter=self._jitter))
            # all_obs = tf.concat([all_obs[None,None,:], obs_at_query_points[:,:,None]], axis=-1)
            all_obs.append(obs_at_query_points)

            acq = acq_with_penalty(pending_points, step)(x_in)
            xnew_idx = tf.argmax(acq, axis=0)
            idx_list = xnew_idx.numpy().tolist()
            pending_points = tf.gather_nd(x_in, [[idx] for idx in idx_list])

        all_obs = tf.stack(all_obs, axis=-1)
        min_across_horizon = tf.reduce_min(all_obs,axis=-1)
        improvement = tf.nn.relu(self._eta - min_across_horizon)
        acq_mc = tf.reduce_mean(improvement, axis=0)  

        return acq_mc[:,None]








# Copyright 2020 The Trieste Contributors
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
"""
This module contains acquisition rules, which choose the optimal point(s) to query on each step of
the Bayesian optimization process.
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Union, cast, overload

import tensorflow as tf

from trieste import logging, types
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.models.interfaces import HasReparamSampler, ProbabilisticModelType
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.types import State, TensorType
from trieste.acquisition.function import BatchMonteCarloExpectedImprovement, ExpectedImprovement
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    GreedyAcquisitionFunctionBuilder,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
    VectorizedAcquisitionFunctionBuilder,
)
from trieste.acquisition.optimizer import (
    AcquisitionOptimizer,
    automatic_optimizer_selector,
    batchify_joint,
    batchify_vectorize,
)
from trieste.acquisition.sampler import ExactThompsonSampler, ThompsonSampler
from trieste.acquisition.utils import select_nth_output

ResultType = TypeVar("ResultType", covariant=True)
""" Unbound covariant type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace, contravariant=True)
""" Contravariant type variable bound to :class:`~trieste.space.SearchSpace`. """


class AcquisitionRule(ABC, Generic[ResultType, SearchSpaceType, ProbabilisticModelType]):
    """
    The central component of the acquisition API.

    An :class:`AcquisitionRule` can produce any value from the search space for this step, and the
    historic data and models. This value is typically a set of query points, either on its own as
    a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some context
    (see e.g. :class:`TrustRegion`). Indeed, to use an :class:`AcquisitionRule` in the main
    :class:`~trieste.bayesian_optimizer.BayesianOptimizer` Bayesian optimization loop, the rule
    must return either a `TensorType` or `State`-ful `TensorType`.

    Note that an :class:`AcquisitionRule` might only support models with specific features (for
    example, if it uses an acquisition function that relies on those features). The type of
    models supported by a rule is indicated by the generic type variable
    class:`ProbabilisticModelType`.
    """

    @abstractmethod
    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> ResultType:
        """
        Return a value of type `T_co`. Typically this will be a set of query points, either on its
        own as a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some
        context (see e.g. :class:`TrustRegion`). We assume that this requires at least models, but
        it may sometimes also need data.

        **Type hints:**
          - The search space must be a :class:`~trieste.space.SearchSpace`. The exact type of
            :class:`~trieste.space.SearchSpace` depends on the specific :class:`AcquisitionRule`.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations for each tag (optional).
        :return: A value of type `T_co`.
        """

    def acquire_single(
        self,
        search_space: SearchSpaceType,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> ResultType:
        """
        A convenience wrapper for :meth:`acquire` that uses only one model, dataset pair.

        :param search_space: The global search space over which the optimization problem
            is defined.
        :param model: The model to use.
        :param dataset: The known observer query points and observations (optional).
        :return: A value of type `T_co`.
        """
        if isinstance(dataset, dict) or isinstance(model, dict):
            raise ValueError(
                "AcquisitionRule.acquire_single method does not support multiple datasets "
                "or models: use acquire instead"
            )
        return self.acquire(
            search_space,
            {OBJECTIVE: model},
            datasets=None if dataset is None else {OBJECTIVE: dataset},
        )


class EfficientGlobalOptimization_qEI(
    AcquisitionRule[TensorType, SearchSpaceType, ProbabilisticModelType]
):
    """Implements the Efficient Global Optimization, or EGO, algorithm."""

    @overload
    def __init__(
        self: "EfficientGlobalOptimization_qEI[SearchSpaceType, ProbabilisticModel]",
        builder: None = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    @overload
    def __init__(
        self: "EfficientGlobalOptimization_qEI[SearchSpaceType, ProbabilisticModelType]",
        builder: (
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
        ),
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    def __init__(
        self,
        builder: Optional[
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
            | VectorizedAcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelVectorizedAcquisitionBuilder[ProbabilisticModelType]
        ] = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        """
        :param builder: The acquisition function builder to use. Defaults to
            :class:`~trieste.acquisition.ExpectedImprovement`.
        :param optimizer: The optimizer with which to optimize the acquisition function built by
            ``builder``. This should *maximize* the acquisition function, and must be compatible
            with the global search space. Defaults to
            :func:`~trieste.acquisition.optimizer.automatic_optimizer_selector`.
        :param num_query_points: The number of points to acquire.
        """

        if num_query_points <= 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if builder is None:
            if num_query_points == 1:
                builder = ExpectedImprovement()
            else:
                raise ValueError(
                    """Need to specify a batch acquisition function when number of query points
                    is greater than 1"""
                )

        if optimizer is None:
            optimizer = automatic_optimizer_selector

        if isinstance(
            builder,
            (
                SingleModelAcquisitionBuilder,
                SingleModelGreedyAcquisitionBuilder,
                SingleModelVectorizedAcquisitionBuilder,
            ),
        ):
            builder = builder.using(OBJECTIVE)

        if num_query_points > 1:  # need to build batches of points
            if isinstance(builder, VectorizedAcquisitionFunctionBuilder):
                # optimize batch elements independently
                optimizer = batchify_vectorize(optimizer, num_query_points)
            elif isinstance(builder, AcquisitionFunctionBuilder):
                # optimize batch elements jointly
                optimizer = batchify_joint(optimizer, num_query_points)
            elif isinstance(builder, GreedyAcquisitionFunctionBuilder):
                # optimize batch elements sequentially using the logic in acquire.
                pass

        self._builder: Union[
            AcquisitionFunctionBuilder[ProbabilisticModelType],
            GreedyAcquisitionFunctionBuilder[ProbabilisticModelType],
            VectorizedAcquisitionFunctionBuilder[ProbabilisticModelType],
        ] = builder
        self._optimizer = optimizer
        self._num_query_points = num_query_points
        self._acquisition_function: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return f"""EfficientGlobalOptimization(
        {self._builder!r},
        {self._optimizer!r},
        {self._num_query_points!r})"""

    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> TensorType:
        """
        Return the query point(s) that optimizes the acquisition function produced by ``builder``
        (see :meth:`__init__`).

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Whether this is required
            depends on the acquisition function used.
        :return: The single (or batch of) points to query.
        """
        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models,
                datasets=datasets,
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models,
                datasets=datasets,
            )

        summary_writer = logging.get_tensorboard_writer()
        step_number = logging.get_step_number()
        greedy = isinstance(self._builder, GreedyAcquisitionFunctionBuilder)

        with tf.name_scope("EGO.optimizer" + "[0]" * greedy):
            points = self._optimizer(search_space, self._acquisition_function)

        if summary_writer:
            with summary_writer.as_default(step=step_number):
                batched_points = tf.expand_dims(points, axis=0)
                values = self._acquisition_function(batched_points)[0]
                if len(values) == 1:
                    logging.scalar(
                        "EGO.acquisition_function/maximum_found" + "[0]" * greedy, values[0]
                    )
                else:  # vectorized acquisition function
                    logging.histogram(
                        "EGO.acquisition_function/maximum_found" + "[0]" * greedy, values
                    )

        if isinstance(self._builder, GreedyAcquisitionFunctionBuilder):
            for i in range(
                self._num_query_points - 1
            ):  # greedily allocate remaining batch elements
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=points,
                    new_optimization_step=False,
                )
                with tf.name_scope(f"EGO.optimizer[{i+1}]"):
                    chosen_point = self._optimizer(search_space, self._acquisition_function)
                points = tf.concat([points, chosen_point], axis=0)

                if summary_writer:
                    with summary_writer.as_default(step=step_number):
                        batched_points = tf.expand_dims(chosen_point, axis=0)
                        values = self._acquisition_function(batched_points)[0]
                        if len(values) == 1:
                            logging.scalar(
                                f"EGO.acquisition_function/maximum_found[{i + 1}]", values[0]
                            )
                        else:  # vectorized acquisition function
                            logging.histogram(
                                f"EGO.acquisition_function/maximum_found[{i+1}]", values
                            )

        return tf.random.shuffle(points)[0:1,:]













# class glasses_ego(AcquisitionFunctionClass):
#     def __init__(
#         self, 
#         model: ProbabilisticModel, 
#         eta: TensorType, 
#         search_space: SearchSpace, 
#         num_lookahead: int,
#         dataset: Dataset,
#         ):
#         r"""
#         :param model: The model of the objective function.
#         :param eta: The "best" observation.
#         :return: The expected improvement function. This function will raise
#             :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
#             greater than one.
#         """
#         self._model = model
#         self._eta = tf.Variable(eta)
#         self._search_space = search_space
#         self._num_lookahead = num_lookahead
#         self._dataset = dataset
#         self._optimizer = generate_random_search_optimizer(100) # 100 samples (make user custom)
#         self._inner_acq_builder = LocalPenalization(self._search_space) # (make user custom)
#         self._inner_acq = self._inner_acq_builder.prepare_acquisition_function(self._model, self._dataset)

#     def update(self, eta: TensorType, dataset: Dataset) -> None:
#         """Update the acquisition function with a new eta value."""
#         self._eta.assign(eta)
#         self._dataset = dataset
#         # update is called for new BO steps
#         # self._inner_acq = self._inner_acq_builder.update_acquisition_function(self._inner_acq, self._model, self._dataset)

#     #@tf.function
#     def __call__(self, x: TensorType) -> TensorType: # [N, 1, d] -> [N, 1]
#         tf.debugging.assert_shapes(
#             [(x, [..., 1, None])],
#             message="This acquisition function only supports batch sizes of one.",
#         )

#         acq_values = tf.constant([], dtype=tf.float64)
#         for x_current in x:
#             self._inner_acq_rule = EfficientGlobalOptimization(num_query_points=self._num_lookahead-1, builder=self._inner_acq_builder, optimizer=generate_random_search_optimizer(1000))
#             self._inner_acq_rule._builder.update_acquisition_function(self._inner_acq, {'OBJECTIVE':self._model}, {'OBJECTIVE':self._dataset}, pending_points=x_current)
#             points_chosen_by_local_penalization = self._inner_acq_rule.acquire_single(self._search_space, self._model, dataset=self._dataset)
#             all_points = tf.concat([x_current, points_chosen_by_local_penalization], axis=0)
#             sampler = self._model.reparam_sampler(100)   # MC samples to evaluate EI, make user custom
#             samples_at_query_points = sampler.sample(all_points)
#             mean = tf.reduce_mean(samples_at_query_points, axis=-3, keepdims=True)  
#             eta_new = tf.squeeze(tf.reduce_min(mean))
#             acq_new = max(self._eta - eta_new, tf.constant([0], dtype=tf.float64))
#             acq_values = tf.concat([acq_values, acq_new], axis=0)
    
#         return acq_values[:,None] # [N, 1]











class glasses_mc(AcquisitionFunctionClass):
    def __init__(
        self, 
        model: ProbabilisticModel, 
        eta: TensorType, 
        search_space: SearchSpace, 
        num_lookahead: int,
        dataset: Dataset,
        ):
        r"""
        :param model: The model of the objective function.
        :param eta: The "best" observation.
        :return: The expected improvement function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        self._model = model
        self._eta = tf.Variable(eta)
        self._search_space = search_space
        self._num_lookahead = num_lookahead
        self._dataset = dataset
        self._optimizer = generate_random_search_optimizer(100) # 100 samples (make user custom)
        self._inner_acq_builder = MonteCarloExpectedImprovement(sample_size=10) # (make user custom)
        self._inner_acq = self._inner_acq_builder.prepare_acquisition_function(self._model, self._dataset)

    def update(self, eta: TensorType, dataset: Dataset) -> None:
        """Update the acquisition function with a new eta value."""
        self._eta.assign(eta)
        self._dataset = dataset
        # update is called for new BO steps
        self._inner_acq = self._inner_acq_builder.update_acquisition_function(self._inner_acq, self._model, self._dataset)

    #@tf.function
    def __call__(self, x: TensorType) -> TensorType: # [N, 1, d] -> [N, 1]
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        acq_values = tf.constant([],dtype=tf.float64)
        for x_current in x:
            def inner_acq_inc_current(next_points: TensorType) -> TensorType: # [N, num_lookahead-1, d] -> [N, 1]
                return self._inner_acq(tf.concat([x_current[None,:,:], next_points],axis=0))

            remainder_of_batch = self._optimizer(self._search_space, inner_acq_inc_current) # [num_lookahead-1, d]
            
            trajectory = tf.stack([x_current, remainder_of_batch], axis=0) # [num_lookahead, d]
            acq = self._inner_acq(trajectory[None, :, :]) # [1,1]
            acq_values = tf.concat([acq_values[None,:,None], acq], axis=0)

        return acq_values # [N, 1]

