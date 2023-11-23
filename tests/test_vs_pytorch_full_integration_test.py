import numpy as np
from typing import List
from typing import List, Tuple, Union, Callable
import unittest

import torch
import torch.nn as nn

import candle
import candle.functions as F


class TestFullIntegrationTestVsPytorch(unittest.TestCase):
            
    def test_model_training(self):
        # This tests initializing a ResNet in both candle and pytorch, then training it AdamW with
        # a learning rate scheduler, and assert that the train/test loss/acc curves are exactly equal     
        
        ATOL = 1e-3

        ## (1) Load data

        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        digits = datasets.load_digits()

        X = digits.images[:, None]  # Add channel dimension

        X_train, X_test, y_train, y_test = train_test_split(  # Split data into 50% train and 50% test subsets
            X, digits.target, test_size=0.5, shuffle=False
        )

        # Standardize data

        train_mean = X_train.mean()
        train_std = X_train.std()
        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        num_features = X_train.shape[1]
        num_classes = len(set(y_train.data))

        X_train = candle.Tensor(X_train)
        y_train = candle.Tensor(y_train)
        X_test = candle.Tensor(X_test)
        y_test = candle.Tensor(y_test)

        ## (2) Define Training Configuration

        class Config:
            # General configs

            ITERATIONS = 100
            BATCH_SIZE = 32
            LEARNING_RATE = 3e-4
            WEIGHT_DECAY = 1e-1

            EVAL_ITERS = 1
            EVAL_BATCH_SIZE = 32 

            # LR scheduler configs

            STEPLR_STEP_SIZE = 1000
            STEPLR_GAMMA = 0.2
            WARMUP_STEPS = 0

            # Model-specific configs

            RESNET_BLOCKS = [
                # (in_channels, out_channels, stride)
                (8, 8, 1),
                (8, 8, 1),

                (8, 16, 1),
                (16, 16, 1),
            ]


        config = Config()

        ## (3) Define Model

        from candle.models.resnet import ResNet

        model = ResNet(num_classes=num_classes,
                       in_channels=X_train.shape[1],
                       resnet_blocks=config.RESNET_BLOCKS)

        model.summary((1, 1, 8, 8))

        optimizer = candle.optimizer.AdamW(model.parameters(),
                                           learning_rate=config.LEARNING_RATE,
                                           weight_decay=config.WEIGHT_DECAY)

        ## (4) Define Pytorch Model and Transfer Weights

        class ResNetTorch(nn.Module):

            def __init__(self,
                         num_classes: int,
                         in_channels: int,
                         resnet_blocks: List[Tuple[int, int, int]]):
                super().__init__()
                self.num_classes = num_classes

                self.conv = nn.Conv2d(in_channels, resnet_blocks[0][0], kernel_size=7, padding=1, stride=1)
                self.batch_norm = nn.BatchNorm2d(resnet_blocks[0][0])
                self.max_pool = nn.MaxPool2d(kernel_size=2)  # Remove MaxPool since MNIST is only 8x8

                self.residual_blocks = nn.ParameterList([
                    ResNetBlockTorch(in_channels, out_channels, stride)
                    for (in_channels, out_channels, stride) in resnet_blocks
                ])

                self.linear = nn.Linear(resnet_blocks[-1][1], num_classes)


            def forward(self, x):
                x = self.conv(x)
                x = self.batch_norm(x)
                x = torch.relu(x)

                x = self.max_pool(x)

                for residual_block in self.residual_blocks:
                    x = residual_block(x)
                    x = torch.relu(x)

                x = x.mean(axis=(2, 3))
                x = self.linear(x)

                return x


        class ResNetBlockTorch(nn.Module):

            def __init__(self,
                         in_channels: int,
                         out_channels: int,
                         stride: int = 1):
                super().__init__()

                self.in_channels = in_channels
                self.out_channels = out_channels
                self.stride = stride

                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

                self.batch_norm1 = nn.BatchNorm2d(out_channels)
                self.batch_norm2 = nn.BatchNorm2d(out_channels)

                if in_channels != out_channels:
                    self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                else:
                    self.res_conv = None


            def forward(self, x):
                x_conv = self.conv1(x)
                x_conv = self.batch_norm1(x_conv)
                x_conv = torch.relu(x_conv)

                x_conv = self.conv2(x_conv)
                x_conv = self.batch_norm2(x_conv)

                if self.res_conv is not None:
                    x = self.res_conv(x)

                x_conv = x + x_conv

                return x_conv

        # Copy torch weights over

        model_torch = ResNetTorch(num_classes=num_classes,
                       in_channels=X_train.shape[1],
                       resnet_blocks=config.RESNET_BLOCKS)

        optimizer_torch = torch.optim.AdamW(model_torch.parameters(),
                                            lr=config.LEARNING_RATE,
                                            weight_decay=config.WEIGHT_DECAY)

        params = model.parameters()
        torch_params = dict(model_torch.named_parameters())

        for p in params:
            p2 = p.replace('.W', '.weight').replace('kernel', 'weight')
            if p2.endswith('.b'):
                p2 = p2[:-2] + '.bias'

            copy_param = torch_params[p2].detach().numpy().copy()

            if len(copy_param.shape) == 1:
                copy_param = copy_param.reshape(params[p].shape)

            elif 'kernel' in p2 or 'weight' in p2:
                copy_param = copy_param.swapaxes(0, 1)

            params[p].data[:] = copy_param

        ## (5) Check output equality

        def ttt(x):
            # to_torch_tensor
            return torch.Tensor(x.data).float()

        # Feed batch in to get activations

        model.train()

        X_batch = X_train[:256]
        y_batch = y_train[:256]

        output = model(X_batch)
        loss = F.cross_entropy_loss(output, y_batch)
        loss.backward()

        model_torch.train()

        model_torch.zero_grad()
        output_torch = model_torch(ttt(X_batch))
        loss_torch = nn.CrossEntropyLoss()(output_torch, ttt(y_batch).long())
        loss_torch.backward()

        assert np.isclose(float(output.sum().data), output_torch.sum().item(), atol=ATOL)
        assert np.isclose(float(loss.data), loss_torch.item(), atol=ATOL)

        assert np.isclose(sum([model.parameters()[p].sum().data.item() for p in model.parameters()]),
                          sum([p.sum().item() for p in model_torch.parameters()]),
                          atol=ATOL)

        # Feed batch in to get activations - eval mode

        model.eval()

        X_batch = X_train[:256]
        y_batch = y_train[:256]

        output = model(X_batch)
        loss = F.cross_entropy_loss(output, y_batch)
        loss.backward()

        model_torch.eval()

        model_torch.zero_grad()
        output_torch = model_torch(ttt(X_batch))
        loss_torch = nn.CrossEntropyLoss()(output_torch, ttt(y_batch).long())
        loss_torch.backward()

        assert np.isclose(float(output.sum().data), output_torch.sum().item(), atol=ATOL)
        assert np.isclose(float(loss.data), loss_torch.item(), atol=ATOL)

        ## (5) Train Model

        def get_random_batch(*tensors, batch_size: int, transforms: List[Callable] = None):
            """Get random batch of data.

            Parameters
            ----------
            tensors
                List of Tensors.
            batch_size
                Size of batches to return.
            transforms
                List with same size as tensors.
                Each element is a list of Callable functions.

            """
            assert len(set([len(t) for t in tensors])) == 1

            if batch_size is None:
                return tensors

            indices = np.random.choice(range(len(tensors[0])), min(batch_size, len(tensors[0])), replace=False)

            items = []
            for (i, tensor) in enumerate(tensors):
                item = tensor[indices]

                if transforms is not None and transforms[i] is not None:
                    for transform in transforms[i]:
                        item = transform(item)

                items.append(item)
            items = tuple(items)

            return items


        def get_loss_and_accuracy(model, X, y, logits: np.array = None):
            """Gets loss and accuracy for a classification model.

            Parameters
            ----------
            model
                Module that outputs logits.
            X
                Features, shape (batch_size, num_features)
            y
                Target, int tensor with shape (batch_size,)

            """
            if logits is None:
                logits = []
                for (X_batch,) in candle.DataLoader(X, batch_size=64, shuffle=False):
                    output = model(X_batch)
                    logits.append(output.data)
                logits = np.concatenate(logits)

            predictions = np.argmax(logits, axis=1)

            loss = float(F.cross_entropy_loss(candle.Tensor(logits), y).data)
            accuracy = 100 * sum(predictions == y.data) / len(y)

            return (loss, accuracy)



        def get_loss_and_accuracy_torch(model, X, y, logits: np.array = None):
            """Gets loss and accuracy for a classification model.

            Parameters
            ----------
            model
                Module that outputs logits.
            X
                Features, shape (batch_size, num_features)
            y
                Target, int tensor with shape (batch_size,)

            """
            if logits is None:
                logits = []
                for (X_batch,) in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size=64, shuffle=False):
                    output = model(X_batch)
                    logits.append(output.data)
                logits = np.concatenate(logits)

            predictions = np.argmax(logits, axis=1)

            loss = float(nn.CrossEntropyLoss()(torch.Tensor(logits), y).data)
            accuracy = 100 * sum(predictions == y.numpy()) / len(y)

            return (loss, accuracy)

        # Create data loaders and data augmentation transforms

        train_transforms = [candle.vision.RandomCrop(8, padding=2)]
        test_transforms = None

        data_loader = candle.DataLoader(X_train, y_train, batch_size=config.BATCH_SIZE,
                                        shuffle=True, drop_last=True, transforms=[train_transforms, None])
        data_iterator = iter(data_loader)

        for iteration in range(config.ITERATIONS):
            model.train()
            model_torch.train()

            try:
                (X_batch, y_batch) = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                (X_batch, y_batch) = next(data_iterator)

            # ------------
            # Candle model
            # ------------

            output = model(X_batch)
            loss = F.cross_entropy_loss(output, y_batch)
            loss.backward()

            optimizer.step()

            # -----------
            # Torch model
            # -----------

            model_torch.zero_grad()
            output_torch = model_torch(ttt(X_batch))
            loss_torch = nn.CrossEntropyLoss()(output_torch, ttt(y_batch).long())
            loss_torch.backward()

            optimizer_torch.step()

            # ----------
            # Assertions
            # ----------

            model.eval()
            model_torch.eval()

            # candle

            (train_loss, train_acc) = get_loss_and_accuracy(model, X_batch, y_batch, output.data)
            test_batch = get_random_batch(X_test, y_test,
                                          batch_size=config.EVAL_BATCH_SIZE,
                                          transforms=[test_transforms, None])
            (test_loss, test_acc) = get_loss_and_accuracy(model, *test_batch)

            # torch

            (train_loss_torch, train_acc_torch) = get_loss_and_accuracy_torch(
                model_torch, ttt(X_batch), ttt(y_batch).long(), output_torch.detach().numpy()
            )

            test_batch = [ttt(t) for t in test_batch]
            test_batch[1] = test_batch[1].long()
            (test_loss_torch, test_acc_torch) = get_loss_and_accuracy_torch(model_torch, *test_batch)

            assert np.isclose(train_loss, train_loss_torch, atol=1e-3)
            assert np.isclose(test_loss, test_loss_torch, atol=1e-3)

            assert np.isclose(train_acc, train_acc_torch, atol=1e-3)
            assert np.isclose(test_acc, test_acc_torch, atol=1e-3)        
            