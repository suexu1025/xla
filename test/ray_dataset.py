import os
import ray
import idx2numpy
from ray.data.preprocessors import TorchVisionPreprocessor
import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms
import tensorflow.io as io

from typing import Union, Dict, Any, Callable, Optional, Tuple, List
#import pickle
import json

def random_split_data(data_size, num_shards, shard_id):
    idx = [*range(0,data_size,1)]
    #idx = np.random.shuffle(idx)
    x = [a.tolist() for a in np.array_split(idx, num_shards)]
    return x[shard_id]

#[ f.path for f in os.scandir(path) if os.path.isdir(f)]
def idx_reader(path):
    arr = idx2numpy.convert_from_file(paths)
    return ray.from_numpy(arr)

def ray_reader(name:str):
    if name.filename('.npy'):
        return ray.data.read_numpy
    elif name.filename('.txt'):
        return ray.data.read_text
    elif name.filename('.png') or name.filename('.jpg') or name.filename('.jpeg') or name.filename('.tiff') or name.filename('.bmp') or name.filename('.gif'):
        return ray.data.read_images
    elif name.filename('.idx3-ubyte'):
        return idx_reader
    else:
        print(name, "type is not support")
    return None
        
def ray_data_loader(datasource: Union[str, List[str]], 
                    filereader: Optional[Callable],
                    transform: Optional[Callable],
                    batch_size:int, 
                    *arg,  **kwargs):

    #readfunc = ray_reader(datasource[0])
    ds = filereader(datasource, *arg, **kwargs)

    ds = ds.map_batches(transform)
    return ds

def ray_dataset_MNIST(path, train, batch_size):
    paths={}
    for dirname, _, filenames in os.walk(path):
        if train:
            paths["images"] = os.path.join(dirname,"train-images-idx3-ubyte")
            paths["labels"] = os.path.join(dirname,"train-labels-idx1-ubyte")    
        else:
            paths["images"] = os.path.join(dirname,"t10k-images-idx3-ubyte")
            paths["labels"] = os.path.join(dirname,"t10k-labels-idx1-ubyte")  


    def to_tensor(batch: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(batch, dtype=torch.float)

        # (B, H, W, C) -> (B, C, H, W)
        #tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.unsqueeze(0)
        # [0., 255.] -> [0., 1.]
        tensor = tensor.div(255)

        return tensor

    def Normalize(batch: torch.Tensor) -> torch.Tensor:
        tensor = torchvision.transforms.Normalize((0.1307,), (0.3081,))(batch)
        return tensor

    def pd_to_tensor(batch: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(batch, dtype=torch.float)

        # (B, H, W, C) -> (B, C, H, W)
        #tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.unsqueeze(0)
        # [0., 255.] -> [0., 1.]
        tensor = tensor.div(255)

        return tensor
    transform=transforms.Compose(
            [transforms.Lambda(to_tensor),
                transforms.Normalize((0.1307,), (0.3081,))])

    arr = idx2numpy.convert_from_file(paths['images'])
    arr_label = idx2numpy.convert_from_file(paths['labels'])
    df = pd.DataFrame({'label': arr_label, 'images': list(arr)}, columns=['label', 'images'])
    ds = ray.data.from_pandas(df)

    # ds = ray.data.from_numpy(arr)
    # labels = ray.data.from_numpy(arr_label)

    preprocessor = TorchVisionPreprocessor(["images"], transform=transform, batched = True)
    ds = preprocessor.transform(ds)
    #ds = ds.map_batches(pd_to_tensor)
    #ds = ds.map_batches(Normalize) 
    return ds

if __name__ == '__main__':
    import time
    test="ray_dataset_cifar"
    if test == "ray_dataset_MNIST":
        ds = ray_dataset_MNIST("/tmp/mnist-data/0/", train=True, batch_size=128)
        epoch = 2
        for idx in range(epoch):
            #ds.random_shuffle()
            tick = time.time()
            for step, data in enumerate(ds.iter_batches(batch_size=128)):
                #image = data["images"]
                #label = data["label"]
                image = torch.tensor(np.array(data["images"].to_list()))
                label = torch.tensor(np.array(data["label"].to_list()))
                print(image.sum())
            toc = time.time() - tick
            print("per peoch time is toc", toc)    
    else:
        data_dir = "/mnt/disks/persist/imagenet"
        with io.gfile.GFile(os.path.join(data_dir, 'imagenetindex_train.json')) as f:
            paths_x = json.load(f)

        #subset = random_split_data(len(paths_x), pt.global_device_count(), xm.get_ordinal())
        subset = random_split_data(len(paths_x), 128, 0)
        paths_x = [ paths_x[i] for i in subset]

        # compond the path
        paths_x = [name.split('train/')[-1] for name in paths_x]
        path = os.path.join(data_dir, "train")
        paths_x = [os.path.join(path, name) for name in paths_x]  

        datapaths = {
            "images": "/tmp/mnist-data/0/train-images-idx3-ubyte",
            "labels": "/tmp/mnist-data/0/train-labels-idx1-ubyte",
        }
        datapaths = {
            "images": "/home/qinwen/data/cifar-100-python/numpy/train-data.npy",
        }

        def to_tensor_slow(batch: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
            tensor = torch.tensor(np.array(batch["image"].to_list())) #torch.as_tensor(batch["image"])

            # (B, H, W, C) -> (B, C, H, W)
            tensor = tensor.permute(0, 3, 1, 2).contiguous().float()
            # [0., 255.] -> [0., 1.]
            tensor = tensor.div(255)
            batch["image"] = list(tensor)
            #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            return batch

        def to_tensor(batch: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
            tensor = torch.tensor(np.array(batch["image"].to_list())) #torch.as_tensor(batch["image"])

            # (B, H, W, C) -> (B, C, H, W)
            tensor = tensor.permute(0, 3, 1, 2).contiguous().float()
            # [0., 255.] -> [0., 1.]
            tensor = tensor.div(255)
            # normalization 
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensor = norm(tensor)
            return {"image": tensor.numpy()}

        #ds = ray_data_loader(datapaths["images"], ray.data.read_binary_files, to_tensor, 128)
        ds = ray_data_loader(paths_x, ray.data.read_images, to_tensor, 128, size=(224, 224), mode="RGB")
        epoch = 2
        for idx in range(epoch):
            #ds.random_shuffle()
            tick = time.time()
            for step, data in enumerate(ds.iter_torch_batches(batch_size=128)):
                image = data["image"]
                #import pdb; pdb.set_trace()
                #label = data["label"]
                #image = torch.tensor(np.array(data["images"].to_list()))
                #label = torch.tensor(np.array(data["label"].to_list()))
                print(image.sum())
            toc = time.time() - tick
            print("per peoch time is toc", toc)  