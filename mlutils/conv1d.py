"""Различные архитектурные элементы сверточных сетей для 1d."""
from enum import Enum
from typing import Callable
from typing import Optional

from keras import layers
import tensorflow as tf


def reznet_layers(x: tf.Tensor, channels: int, bach_norm: bool = False):
    """Стандартный набор слоев в RezNet v2."""
    if bach_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(channels, kernel_size=3, strides=1, padding="same")(x)
    if bach_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(channels, kernel_size=3, strides=1, padding="same")(x)
    return x


def reznet_bottleneck_layers(x: tf.Tensor, channels: int, bach_norm: bool = False):
    """Набор слоев в RezNet v2 с бутылочным горлышком - обычно для более глубоких сетей."""
    if bach_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(channels // 4, kernel_size=1, strides=1, padding="same")(x)
    if bach_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(channels // 4, kernel_size=3, strides=1, padding="same")(x)
    if bach_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(channels, kernel_size=1, strides=1, padding="same")(x)
    return x


def densenet_layers(x: tf.Tensor, channels: int, bach_norm: bool = False):
    """Стандартный набор слоев в DenseNet."""
    if bach_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(4 * channels, kernel_size=1, strides=1, padding="same")(x)
    if bach_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(channels, kernel_size=3, strides=1, padding="same")(x)
    return x


class Layers(Enum):
    """Базовые слои для построения сети."""
    COMMON = reznet_layers
    BOTTLENECK = reznet_bottleneck_layers
    DENSE = densenet_layers


def se_block(x: tf.Tensor, ratio: Optional[int] = None) -> tf.Tensor:
    """Squeeze-and-Excitation block для 1D сверточных сетей.

    На основе https://arxiv.org/pdf/1709.01507.pdf

    :param x:
        Тензор из 1D сверточной сети вида (None, time_steps, channels).
    :param ratio:
        Кратность от 0.0 до 1ю0 сокращения количества каналов (squeeze) - по умолчанию уменьшается до корня от
        количества каналов.
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


def reznet_link(
        x: tf.Tensor,
        channels: int,
        layers_func: Callable[[tf.Tensor, int, bool], tf.Tensor],
        bach_norm: bool = False) -> tf.Tensor:
    """Остаточное соединение из RezNet.

    :param x:
        Тензор из 1D сверточной сети вида (None, time_steps, channels).
    :param channels:
        На входе первого блока.
    :param layers_func:
        Функция с основными слоями преобразования сети.
    :param bach_norm:
        Нужна ли бач-нормализация основных слоев.
    :return:
        Результат применения основного преобразования и скипа.
    """
    y = x
    y = layers_func(y, channels, bach_norm)
    y = layers.add([x, y])
    return y


def densenet_link(
        x: tf.Tensor,
        channels: int,
        layers_func: Callable[[tf.Tensor, int, bool], tf.Tensor],
        bach_norm: bool = False) -> tf.Tensor:
    """Остаточное соединение из DenseNet.

    :param x:
        Тензор из 1D сверточной сети вида (None, time_steps, channels).
    :param channels:
        На входе первого блока.
    :param layers_func:
        Функция с основными слоями преобразования сети.
    :param bach_norm:
        Нужна ли бач-нормализация основных слоев.
    :return:
        Результат применения основного преобразования и скипа.
    """
    y = x
    y = layers_func(y, channels, bach_norm)
    y = layers.concatenate([x, y])
    return y


class Links(Enum):
    """Тип остаточного соединения."""
    REZNET = reznet_link
    DENSENET = densenet_link


def make_net(
        x: tf.Tensor,
        blocks: int,
        link: Links,
        channels: int,
        layers_type: Layers,
        bach_norm: bool = False,
        se: bool = False
) -> tf.Tensor:
    """Построение сети по мотивам основных сверточных архитектур.

    :param x:
        Тензор из 1D сверточной сети вида (None, time_steps, channels).
    :param blocks:
        Количество блоков.
    :param link:
        Тип остаточного соединения.
    :param channels:
        Количество каналов.
    :param layers_type:
        Тип слоев в блоке.
    :param bach_norm:
        Нужна ли бач-нормализация основных слоев.
    :param se:
        Нужно ли взвесить блок с помощью squeeze-and-Excitation.
    :return:
        Результирующий тензор.
    """
    y = layers.Conv1D(
        filters=channels,
        kernel_size=1,
        strides=1,
        padding="same",
        activation=None
    )(x)
    for _ in range(blocks):
        y = layers_type.value(y, channels, bach_norm)
        if se:
            y = se_block(y)
        y = link.value(y)
    return y
