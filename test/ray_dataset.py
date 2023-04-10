import os
import ray
import idx2numpy
from ray.data.preprocessors import TorchVisionPreprocessor
import numpy as np
import pandas as pd

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

    preprocessor = TorchVisionPreprocessor(["images"], transform=transform)
    ds = preprocessor.transform(ds)
    return ds
