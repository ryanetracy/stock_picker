
"""
these functions handle the bulk of the training for the torch NN models
"""

import torch 
from torch import nn
from timeit import default_timer as timer
from typing import Dict


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> float:
    """handle model training.

    Args:
        model (torch.nn.Module): model to train.
        dataloader (torch.utils.data.DataLoader): dataloader for training data.
        loss_fn (torch.nn.Module): loss function.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        float: training loss.
    """
    train_loss = 0

    model.train()

    for X_batch, y_batch in dataloader:
        y_pred = model(X_batch).squeeze(-1)
        loss = torch.sqrt(loss_fn(y_pred, y_batch))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_loss /= len(dataloader)

    return train_loss

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> float:
    """handle model testing.

    Args:
        model (torch.nn.Module): model to train.
        dataloader (torch.utils.data.DataLoader): dataloader for testing data.
        loss_fn (torch.nn.Module): loss function.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        float: testing loss.
    """
    test_loss = 0

    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch in dataloader:
            test_preds = model(X_batch).squeeze(-1)
            loss = torch.sqrt(loss_fn(test_preds, y_batch))

            test_loss += loss

        test_loss /= len(dataloader)

    return test_loss

def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int
) -> Dict[torch.tensor, torch.tensor]:
    """run the full model training and testing pipeline.

    Args:
        model (torch.nn.Module): model to train.
        train_dataloader (torch.utils.data.DataLoader): dataloader for training.
        test_dataloader (torch.utils.data.DataLoader): dataloader for testing.
        loss_fn (torch.nn.Module): loss function.
        optimizer (torch.optim.Optimizer): optimizer.
        epochs (int): number of training epochs.

    Returns:
        Dict[torch.tensor, torch.tensor]: train/test results returned as a 
            dictionary.
    """
    res = {
        "train_loss": [],
        "test_loss": []
    }

    start_time = timer()
    for epoch in range(epochs):
        epoch_start = timer()

        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

        print(
            f"epoch: {epoch}\n",
            f"train loss: {train_loss:.5f}\n",
            f"test_loss: {test_loss:.5f}\n",
            f"run time: {timer() - epoch_start}\n\n"
        )

        res["train_loss"].append(train_loss)
        res["test_loss"].append(test_loss)

    print(f"total run time: {timer() - start_time}")

    return res
