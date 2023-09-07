import os.path
from pathlib import Path
import os
from typing import List

import bisect
import shutil
import imagesize
import cv2
import numpy as np

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

def fd_images(imgdir, savedir):
    import shutil
    imgdir = Path(imgdir)
    savedir = Path(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not imgdir.is_dir():
        raise RuntimeError(f'Invalid directory "{imgdir}"')

    img_paths = collect_images_all(imgdir)
 
    idx = 0
    print('==========================')
    print('========Select Pic========')
    print('==========================')
    for path in img_paths:
        idx += 1
        imgname = os.path.basename(path)
        shutil.copyfile(path, os.path.join(savedir, 'coco_' + imgname))
        print(f'select {idx:04d} -> ' + 'coco' + imgname)

        
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
    for path in namepath:
        idx += 1
        imgname = os.path.basename(path)
        xxxname = os.path.splitext(imgname)[-1]
        name = os.path.splitext(imgname)[0]
        if xxxname != '.png':
            img = cv2.imread(path)
            
            img = np.array((img)).astype(('float64'))
            height, width, channel = img.shape
            ### adding unifor noise
            noise = np.random.uniform(0, 1, (height, width, channel)).astype('float32')
            img += noise
            img = img.astype('uint8')
            img = cv2.resize(img, dsize=((int(width//2), int(height//2))), interpolation=cv2.INTER_CUBIC)
  
            cv2.imwrite(os.path.join(savedir, 'coco_imagenet_' + name + '.png'), img)
            print(f'select {idx:04d} -> ' + 'coco_imagenet_' + name + '.png')
        else:
            shutil.copyfile(path, os.path.join(savedir, imgname))
            print(f'select {idx:04d} -> ' + imgname)

if __name__ == '__main__':
    inputimagedir = './org'   ## original image dataset
    tmpdir = './train'       
    select_n_images(inputimagedir, tmpdir, 100000, 512)