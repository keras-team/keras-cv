# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import FusedConvolution


class SPPCSPC(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        n=1,
        groups=1,
        e=0.5,
        kernel_size=(5, 9, 13),
        deploy=False,
        **kwargs,
    ):
        super(SPPCSPC, self).__init__(**kwargs)
        c_ = int(2 * filters * e)
        self.filters = filter
        self.n = n
        self.groups = groups
        self.e = e
        self.kernel_size = kernel_size
        self.deploy = deploy
        self.cv1 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv2 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv3 = FusedConvolution(c_, 3, 1, deploy=deploy, groups=groups)
        self.cv4 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv5 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv6 = FusedConvolution(c_, 3, 1, deploy=deploy, groups=groups)
        self.cv7 = FusedConvolution(filters, 1, 1, deploy=deploy)
        self.m = [
            tf.keras.layers.MaxPooling2D(pool_size=n, strides=1, padding="same")
            for n in kernel_size
        ]

    def call(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        spp_output = tf.concat([x1] + [m(x) for m in self.m], axis=-1)
        csp_inp1 = self.cv6(self.cv5(spp_output))
        csp_inp2 = self.cv2(x)
        return self.cv7(tf.concat([csp_inp1, csp_inp2], axis=-1))

    def config(self):
        config = super(SPPCSPC, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "n": self.n,
                "groups": self.groups,
                "e": self.e,
                "kernel_size": self.kernel_size,
            }
        )
        return config
