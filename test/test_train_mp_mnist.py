import args_parse
from torch_xla.experimental import pjrt

MODEL_OPTS = {
    '--ddp': {
        'action': 'store_true',
    },
    '--pjrt_distributed': {
        'action': 'store_true',
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/mnist-data',
    batch_size=128,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=18,
    opts=MODEL_OPTS.items(),
)


import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.distributed.xla_backend

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


class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_mnist(flags, **kwargs):
  if flags.pjrt_distributed:
    import torch_xla.experimental.pjrt_backend
    dist.init_process_group('xla', init_method='pjrt://')
  elif flags.ddp:
    dist.init_process_group(
        'xla', world_size=xm.xrt_world_size(), rank=xm.get_ordinal())

  torch.manual_seed(1)

  if flags.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(flags.batch_size, 1, 28,
                          28), torch.zeros(flags.batch_size,
                                           dtype=torch.int64)),
        sample_count=60000 // flags.batch_size // xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(flags.batch_size, 1, 28,
                          28), torch.zeros(flags.batch_size,
                                           dtype=torch.int64)),
        sample_count=10000 // flags.batch_size // xm.xrt_world_size())
  elif flags.loader=="torch_loader":
    train_dataset = datasets.MNIST(
        os.path.join(flags.datadir, str(xm.get_ordinal())),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST(
        os.path.join(flags.datadir, str(xm.get_ordinal())),
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
    train_sampler = None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=flags.batch_size,
        sampler=train_sampler,
        drop_last=flags.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=flags.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags.batch_size,
        drop_last=flags.drop_last,
        shuffle=False,
        num_workers=flags.num_workers)
  else:
    train_loader = ray_dataset_MNIST(os.path.join("/tmp/mnist-data/", str(xm.get_ordinal())), train=True, batch_size=32)
    test_loader = ray_dataset_MNIST(os.path.join("/tmp/mnist-data/", str(xm.get_ordinal())), train=False, batch_size=32)

  # Scale learning rate to num cores
  lr = flags.lr * xm.xrt_world_size()

  device = xm.xla_device()
  model = MNIST().to(device)

  # Initialization is nondeterministic with multiple threads in PjRt.
  # Synchronize model parameters across replicas manually.
  if pjrt.using_pjrt():
    pjrt.broadcast_master_param(model)

  if flags.ddp:
    model = DDP(model, gradient_as_bucket_view=True)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
  loss_fn = nn.NLLLoss()

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    loader.random_shuffle()
    device = xm.xla_device()
    for step, batch in  enumerate(loader.iter_batches(batch_size=flags.batch_size)):
    #for step, (data, target) in enumerate(loader):
      data = torch.tensor(np.array(batch["images"].to_list()))
      target = torch.tensor(np.array(batch["label"].to_list()))
      data = xm.send_cpu_data_to_device(data, device)
      data.to(device)

      target = xm.send_cpu_data_to_device(target, device)
      target.to(device)

      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      if flags.ddp:
        optimizer.step()
      else:
        xm.optimizer_step(optimizer)
      tracker.add(flags.batch_size)
      if step % flags.log_steps == 0:
        xm.add_step_closure(
            _train_update,
            args=(device, step, loss, tracker, epoch, writer),
            run_async=flags.async_closures)

  def test_loop_fn(loader):
    total_samples = 0
    correct = 0
    model.eval()
    device = xm.xla_device()
    for batch in loader.iter_batches(batch_size=flags.batch_size):
      data = torch.tensor(batch["images"])
      target = torch.tensor(batch["label"])
      data = xm.send_cpu_data_to_device(data, device)
      data.to(device)

      target = xm.send_cpu_data_to_device(target, device)
      target.to(device)     
    #for data, target in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  #train_device_loader = pl.MpDeviceLoader(train_loader, device)
  #test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(1, flags.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    accuracy = test_loop_fn(test_loader)
    xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
        epoch, test_utils.now(), accuracy))
    max_accuracy = max(accuracy, max_accuracy)
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={'Accuracy/test': accuracy},
        write_xla_metrics=True)
    if flags.metrics_debug:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_mnist(flags)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)
  if accuracy < flags.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  flags.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
  #_mp_fn(0, FLAGS)
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
