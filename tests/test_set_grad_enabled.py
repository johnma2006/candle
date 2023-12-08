import numpy as np
import unittest

import candle


class TestSetGradEnabled(unittest.TestCase):
    
    def test_set_grad_enabled(self):
        candle.set_grad_enabled(True)
        assert candle.is_grad_enabled()

        with candle.set_grad_enabled(False):
            assert not candle.is_grad_enabled()    
        assert candle.is_grad_enabled()

        with candle.set_grad_enabled(True):
            assert candle.is_grad_enabled()    
        assert candle.is_grad_enabled()

        candle.set_grad_enabled(True)
        assert candle.is_grad_enabled()

        candle.set_grad_enabled(False)
        assert not candle.is_grad_enabled()

        candle.set_grad_enabled(True)
        assert candle.is_grad_enabled()

        with candle.no_grad():
            assert not candle.is_grad_enabled()
        assert candle.is_grad_enabled()

        candle.no_grad()
        assert candle.is_grad_enabled()
        
        candle.set_grad_enabled(True)
        
        
    def test_pointers_with_no_grad(self):
        x = candle.Parameter([1, 2, 3])
        y = 2 * x
        assert y.operation is not None
        assert y.operation.output is not None

        with candle.no_grad():
            x = candle.Parameter([1, 2, 3])
            y = 2 * x
            assert y.operation is None

        x = candle.Parameter([1, 2, 3])
        y = 2 * x
        assert y.operation is not None
        assert y.operation.output is not None

        candle.no_grad()

        x = candle.Parameter([1, 2, 3])
        y = 2 * x
        assert y.operation is not None
        assert y.operation.output is not None

        candle.set_grad_enabled(False)

        x = candle.Parameter([1, 2, 3])
        y = 2 * x
        assert y.operation is None

        candle.set_grad_enabled(True)

        x = candle.Parameter([1, 2, 3])
        y = 2 * x
        assert y.operation is not None
        assert y.operation.output is not None

        candle.set_grad_enabled(True)


    def test_inference_with_no_grad(self):
        model = candle.models.resnet.ResNet(n_classes=10,
                                            in_channels=3,
                                            resnet_blocks=[
                                                (16, 16, 1),
                                                (16, 16, 1),
                                                (16, 16, 1),

                                                (16, 32, 2),
                                                (32, 32, 1),
                                                (32, 32, 1),

                                                (32, 64, 2),
                                                (64, 64, 1),
                                                (64, 64, 1),
                                            ])

        x = candle.Tensor(np.random.normal(size=(256, 3, 32, 32)))
        output1 = model(x)

        with candle.no_grad():
            output2 = model(x)

        assert np.all(output1.data == output2.data)

        