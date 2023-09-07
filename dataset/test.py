import os.path
from pathlib import Path
import os
from typing import List

import bisect
import shutil
import imagesize
import cv2
import numpy as np
import multiprocessing

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

# def select_n_images(imgdir, savedir):
#     """
#     :param imgdir: input image dir
#     :param savedir: seleted image savingdir
#     :param n: the largest n images in the imgdir
#     :return:
#     """
#     import shutil
#     imgdir = Path(imgdir)
#     savedir = Path(savedir)
#     if not os.path.exists(savedir):
#         os.makedirs(savedir)

#     if not imgdir.is_dir():
#         raise RuntimeError(f'Invalid directory "{imgdir}"')

#     img_paths = collect_images_all(imgdir)
 
#     idx = 0
#     print('==========================')
#     print('========Select Pic========')
#     print('==========================')
#     for path in img_paths:
#         idx += 1
#         imgname = os.path.basename(path)
#         shutil.copyfile(path, os.path.join(savedir, 'coco_' + imgname))
#         print(f'select {idx:04d} -> ' + 'coco' + imgname)



def process_image(path, savedir, idx):
    imgname = os.path.basename(path)
    xxxname = os.path.splitext(imgname)[-1]
    name = os.path.splitext(imgname)[0]

    if xxxname != '.png':
        img = cv2.imread(path)
        img = np.array(img).astype('float64')
        height, width, channel = img.shape
        
        noise = np.random.uniform(0, 1, (height, width, channel)).astype('float32')
        img += noise
        img = img.astype('uint8')
        ds_flag = 0

        if min(width, height) > 1024:
            img = cv2.resize(img, dsize=(int(width // 2), int(height // 2)), interpolation=cv2.INTER_CUBIC)
            ds_flag = 1

        cv2.imwrite(os.path.join(savedir, 'coco_imagenet_' + name + '.png'), img)
        print(f'select {idx:04d} -> ' + 'coco_imagenet_' + name + '.png' + f' ===> downscale: {ds_flag}')
    else:
        shutil.copyfile(path, os.path.join(savedir, imgname))
        print(f'select {idx:04d} -> ' + imgname)



        
def select_n_images(imgdir, savedir, n, minsize):
    """
    :param imgdir: input image dir
    :param savedir: seleted image savingdir
    :param n: the largest n images in the imgdir
    :return:
    """
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

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    for idx, path in enumerate(namepath):
        pool.apply_async(process_image, (path, savedir, idx))

    pool.close()
    pool.join()

if __name__ == '__main__':
    inputimagedir = './org'   ## original image dataset
    tmpdir = './dataset_largerthan512_num100000'
    select_n_images(inputimagedir, tmpdir, 100000, 512)