"""Различные архитектурные элементы сверточных сетей."""
from typing import Optional

from keras import layers
import tensorflow as tf


def se_block_1d(x: tf.Tensor, ratio: Optional[int] = None) -> tf.Tensor:
    """Squeeze-and-Excitation block для 1D сверточных сетей.

    На основе https://arxiv.org/pdf/1709.01507.pdf

    :param x:
        Тензор из 1D сверточной сети вида (None, time_steps, channels).
    :param ratio:
        Кратность сокращения количества каналов (squeeze) - по умолчанию уменьшается до корня от количества каналов.
    :return:
        Взвешенный с помощью squeeze-and-Excitation входной тензор.
    """
    channels = int(x.shape[-1])
    if ratio is None:
        bottleneck = int(channels ** 0.5)
    else:
        bottleneck = channels // ratio
    y = layers.GlobalAveragePooling1D()(x)
    y = layers.Dense(bottleneck, activation="relu")(y)
    y = layers.Dense(channels, activation="sigmoid")(y)
    return layers.multiply([x, y])
