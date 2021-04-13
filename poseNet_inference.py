"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import numpy as np
from PIL import Image
import oneflow as flow
import oneflow.typing as tp
import lenet_train

poseLabels = ["frontal", "profile45", "profile75", "upward", "downward"]

def load_image(image_path=''):
    print(image_path)
    im = Image.open(image_path)
    im = im.resize((28, 28))
    im = im.convert('RGB')  # 有的图像是单通道的，不加转换会报错
    im = np.array(im).astype('float32')
    im = (im - 0) / 255
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')



@flow.global_function("predict", flow.function_config())
def InferenceNet(images: tp.Numpy.Placeholder((1, 3, 28, 28), dtype=flow.float)) -> tp.Numpy:
    logits = lenet_train.lenet(images, train=False)
    predictions = flow.nn.softmax(logits)
    return predictions


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.load("./poselenet_models3")

    image = load_image("./testFace/0-3.jpg")
    predictions = InferenceNet(image)
    labels = predictions[0]
    label = np.argmax(labels)
    print(poseLabels[label])
