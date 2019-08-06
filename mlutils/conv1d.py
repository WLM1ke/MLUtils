"""Различные архитектурные элементы сверточных сетей для 1d."""
from typing import Optional

from keras import layers
import tensorflow as tf


def reznet_block(x: tf.Tensor, bach_norm: bool = False) -> tf.Tensor:
    """Обычный блок RezNet для неглубоких сетей.

    :param x:
        Тензор из 1D сверточной сети вида (None, time_steps, channels).
    :param bach_norm:
        Нужна ли бач-нормализация.
    :return:
        Исходящий 1D сверточной сети вида (None, time_steps, channels).
    """
    channels = int(x.shape[-1])
    y = x

    if bach_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(channels, kernel_size=3, strides=1, padding="same")(y)

    if bach_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(channels, kernel_size=3, strides=1, padding="same")(y)

    y = layers.add([y, x])
    return y


def reznet_bottleneck_block(x: tf.Tensor, bach_norm: bool = False) -> tf.Tensor:
    """Обычный узкий и длинный блок RezNet для глубоких сетей.

    По вычислительной сложности при числе каналов в 4 раза больше обычного блока имеет сопоставимое число параметров и
    вычислительную сложность.

    :param x:
        Тензор из 1D сверточной сети вида (None, time_steps, channels).
    :param bach_norm:
        Нужна ли бач-нормализация.
    :return:
        Исходящий 1D сверточной сети вида (None, time_steps, channels).
    """
    channels = int(x.shape[-1])
    y = x

    if bach_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(channels // 4, kernel_size=1, strides=1, padding="same")(y)

    if bach_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(channels // 4, kernel_size=3, strides=1, padding="same")(y)

    if bach_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(channels, kernel_size=1, strides=1, padding="same")(y)

    y = layers.add([y, x])
    return y


def se_block(x: tf.Tensor, ratio: Optional[int] = None) -> tf.Tensor:
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
