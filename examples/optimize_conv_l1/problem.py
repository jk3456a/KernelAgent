# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# KernelBench level1 #50: standard 2D convolution (square input, square kernel).
# AlexNet first conv: Conv2d(3 -> 96, k=11, stride=4, padding=2) on 256x3x224x224.
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


batch_size = 256
num_classes = 1000


def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]
