
import tensorflow_datasets as tfds
import tensorflow as tf
import torch
import time
from torchvision import datasets, transforms

class TorchConvertDataLoader:
  def __init__(self, tf_dl: tf.data.Dataset, transform=None, *args, **kwargs):
    self._iter = tf_dl.as_numpy_iterator()
    self.tf_dl = tf_dl
    self.transform = transform
    super().__init__(*args, **kwargs)

  def __iter__(self):
    return self

  def __next__(self):
    image, label = next(self._iter)
    if self.transform:
        image = self.transforms(image)
    return torch.from_numpy(image), torch.from_numpy(label)

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    mean = 0.1307
    std = 0.3081
    image = (image - mean)/std
    return image, label

def tfdata_MNIST(path, train, batch_size, epoch = 1):
    if path is None:
        ds_builder = tfds.builder('mnist')
        ds_builder.download_and_prepare()

    def process_example(example):
        return example['image'], example['label']
    if train:
        ds = ds_builder.as_dataset(split='train')
    else:
        ds = ds_builder.as_dataset(split='test')

    ds = ds.map(process_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(scale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(10)
    ds.repeat(epoch)
    ds = TorchConvertDataLoader(ds)
    return iter(ds)

if __name__ == '__main__':
    epochs = 1
    ds = tfdata_MNIST(None, train = False, batch_size = 128, epoch=epochs)
    import time
    for idx in range(epochs):
        tic = time.time()
        total = 0
        for step, _ in enumerate(ds):
            image, label = _
            total = total + 1
            pass
        toc = time.time() - tic
        print("total time is ", toc/epochs)
        print("batches, ", total)