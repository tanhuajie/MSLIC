'''
    modified by Tanhuajie
'''

# Copyright 2023 Bytedance Inc.
# All rights reserved.
# Licensed under the BSD 3-Clause Clear License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://choosealicense.com/licenses/bsd-3-clause-clear/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os.path
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import os
from typing import List

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]

def collect_images_all(rootpath: str) -> List[str]:
    image_paths = []

    def search_images(directory):
        for root, _, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS:
                    image_paths.append(os.path.join(root, f))
                    print(os.path.join(root, f))

    search_images(rootpath)
    return image_paths

def preprocessing(imgdir, savedir):
    """
    :param imgdir: input ILSVRC largest 8000 images
    :param savedir: the proprecessed image save dir
    :return:
    """
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not imgdir.is_dir():
        raise RuntimeError(f'Invalid directory "{imgdir}"')

    img_paths = collect_images(imgdir)
    idx = 0
    print('==========================')
    print('=========Add Noise========')
    print('==========================')
    for imgpath in img_paths:
        idx += 1
        img = cv2.imread(imgpath)
        img = np.array((img)).astype(('float64'))
        height, width, channel = img.shape
        ### adding unifor noise
        noise = np.random.uniform(0, 1, (height, width, channel)).astype('float32')
        img += noise
        img = img.astype('uint8')
        if min(width, height)>960:
            img = cv2.resize(img, dsize=((int(width//2), int(height//2))), interpolation=cv2.INTER_CUBIC)
        name = os.path.splitext(os.path.basename(imgpath))[0]
        cv2.imwrite(os.path.join(savedir, name + '.png'), img)
        print(f'exec {idx:04d} -> ' + name + '.png')

def select_n_images(imgdir, savedir, n, minsize):
    """
    :param imgdir: input image dir
    :param savedir: seleted image savingdir
    :param n: the largest n images in the imgdir
    :return:
    """
    import bisect
    import shutil
    import imagesize
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not imgdir.is_dir():
        raise RuntimeError(f'Invalid directory "{imgdir}"')

    img_paths = collect_images_all(imgdir)
    sizepath = []
    namepath = []
    for imgpath in img_paths:
        width, height = imagesize.get(imgpath)
        if min(width, height)>=minsize:
            size = width*height
            loc = bisect.bisect_left(sizepath, size)
            sizepath.insert(loc, size)
            namepath.insert(loc, imgpath)
            if len(sizepath)>n:
                sizepath.pop(0)
                namepath.pop(0)
    idx = 0
    print('==========================')
    print('========Select Pic========')
    print('==========================')
    for path in namepath:
        idx += 1
        imgname = os.path.basename(path)
        shutil.copyfile(path, os.path.join(savedir, imgname))
        print(f'select {idx:04d} -> ' + imgname)


if __name__ == '__main__':
    inputimagedir = './ImageNet'   ## original image dataset
    tmpdir = './tmp'               ## temporary image folder
    savedir = './train_dataset'    ## preprocessed image folder
    select_n_images(inputimagedir, tmpdir, 8000, 480) ## select 8000 images from ImageNet training dataset (larger than 480*480, and the 8000th largest images) 
    preprocessing(tmpdir, savedir)