import tensorflow as tf

from trieste.models.gpflux import (
    DeepGaussianProcess,
    build_vanilla_deep_gp,
)
from trieste.acquisition.rule import (
    EfficientGlobalOptimization,
)
from trieste.acquisition.function import (
    NegativeModelTrajectory
)
from trieste.models.optimizer import KerasOptimizer
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.space import SearchSpace
from trieste.data import Dataset


def build_vanilla_dgp_model(
        data: Dataset,
        search_space: SearchSpace,
        num_layers: int = 2,
        num_inducing: int = 100,
        learn_noise: bool = False,
        epochs: int = 400
):
    if learn_noise:
        noise_variance = 1e-3
    else:
        noise_variance = 1e-5

    dgp = build_vanilla_deep_gp(data, search_space, num_layers, num_inducing,
                                likelihood_variance=noise_variance, trainable_likelihood=learn_noise)

    acquisition_function = NegativeModelTrajectory()
    acquisition_rule = EfficientGlobalOptimization(acquisition_function, optimizer=generate_continuous_optimizer(1000))

    batch_size = 1000

    def scheduler(epoch: int, lr: float) -> float:
        if epoch == epochs // 2:
            return lr * 0.1
        else:
            return lr

    keras_optimizer = tf.optimizers.Adam(0.01, beta_1=0.5, beta_2=0.5)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 0,
        "shuffle": False,
        "callbacks": [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    }
    optimizer = KerasOptimizer(keras_optimizer, fit_args)

    return (
        DeepGaussianProcess(dgp, optimizer, continuous_optimisation=False),
        acquisition_rule
    )


def normalize(x, mean=None, std=None):
    if mean is None:
        mean = tf.math.reduce_mean(x, 0, True)
    if std is None:
        std = tf.math.sqrt(tf.math.reduce_variance(x, 0, True))
    return (x - mean) / std, mean, std