import numpy as np
from typing import List, Callable
import candle
from candle import functions as F
from candle.optimizer import AdamW


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


def get_predictions(model, X):
    """Gets predictions for a classification model.
    
    Parameters
    ----------
    model
        Module that outputs logits.
    X
        Features, shape (batch_size, num_features)
        
    """
    predictions = []
    for (X_batch,) in candle.DataLoader(X, batch_size=64, shuffle=False):
        output = model(X_batch)
        prediction_batch = np.argmax(output.data, axis=1)
        predictions.append(prediction_batch)
        
    predictions = np.concatenate(predictions)
        
    return predictions


def get_gradients(param_names: List[str], model):
    
    grad_by_layer = {name: model.parameters()[name].grad.flatten() for name in param_names}
    gradients = np.concatenate(list(grad_by_layer.values()))
    grad_norm_by_layer = {layer_name: np.linalg.norm(grad) for (layer_name, grad) in grad_by_layer.items()}
    
    return (gradients, grad_by_layer, grad_norm_by_layer)


def get_parameters(param_names: List[str], model):
    
    param_by_layer = {name: model.parameters()[name].data.flatten() for name in param_names}
    parameters = np.concatenate(list(param_by_layer.values()))
    param_norm_by_layer = {layer_name: np.linalg.norm(param) for (layer_name, param) in param_by_layer.items()}
    
    return (parameters, param_by_layer, param_norm_by_layer)


def get_adam_updates(param_names: List[str], optimizer: AdamW):
    
    upd_by_layer = {name: optimizer.compute_update(optimizer.momentum[name],
                                                   optimizer.variance[name],
                                                   optimizer.t)
                    for name in param_names}
    updates = np.concatenate([i.flatten() for i in upd_by_layer.values()])
    upd_norm_by_layer = {layer_name: np.linalg.norm(update) for (layer_name, update) in upd_by_layer.items()}
    
    return (updates, upd_by_layer, upd_norm_by_layer)


def get_adam_mom(param_names: List[str], optimizer: AdamW):
    
    assert isinstance(optimizer, AdamW)
    adam_mom_by_layer = {name: optimizer.momentum[name] / (1 - optimizer.betas[0] ** optimizer.t)
                         for name in param_names}
    adam_mom = np.concatenate([i.flatten() for i in adam_mom_by_layer.values()])
    adam_mom_norm_by_layer = {layer_name: np.linalg.norm(mom) for (layer_name, mom) in adam_mom_by_layer.items()}

    return (adam_mom, adam_mom_by_layer, adam_mom_norm_by_layer)


def get_adam_var(param_names: List[str], optimizer: AdamW):
    
    assert isinstance(optimizer, AdamW)
    adam_var_by_layer = {name: optimizer.variance[name] / (1 - optimizer.betas[1] ** optimizer.t)
                         for name in param_names}
    adam_var = np.concatenate([i.flatten() for i in adam_var_by_layer.values()])
    adam_var_norm_by_layer = {layer_name: np.linalg.norm(var) for (layer_name, var) in adam_var_by_layer.items()}

    return (adam_var, adam_var_by_layer, adam_var_norm_by_layer)
