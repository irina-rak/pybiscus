import torch
from rich.progress import track, Progress

torch.backends.cudnn.enabled = True


def train_loop(fabric, net, trainloader, optimizer, epochs: int, verbose=False):
    """Train the network on the training set."""

    net.train()

    if not optimizer:
        optimizer = None
    elif isinstance(optimizer, list) and len(optimizer) == 1:
        optimizer = optimizer[0]
    
    # Check if the model uses manual optimization (Lightning GANs, etc.)
    uses_manual_optimization = hasattr(net, 'automatic_optimization') and not net.automatic_optimization
    
    with Progress() as progress:
        train_task = progress.add_task("[cyan]Training...", total=len(trainloader))
        for epoch in range(epochs):
            results_epoch = {
                key: torch.tensor(0.0, device=net.device)
                for key in net.signature.__required_keys__
            }
            progress.update(train_task, description=f"Training... Epoch {epoch + 1}/{epochs}")
            
            for batch_idx, batch in enumerate(trainloader):
                progress.update(train_task, advance=1)
                results = net.training_step(batch, batch_idx)
                loss = results["loss"]

                # Only handle optimization if the model doesn't use manual optimization
                if not uses_manual_optimization and optimizer is not None:
                    optimizer.zero_grad()
                    fabric.backward(loss)
                    optimizer.step()

                for key in results_epoch.keys():
                    value = results[key]

                    # hardening code --- begin ---
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, device=net.device)

                    if value.shape != results_epoch[key].shape:
                        value = value.reshape(results_epoch[key].shape)
                    # hardening code --- end ---

                    results_epoch[key] += value
                    # Reset the train_task count
            progress.reset(train_task, total=len(trainloader))

        for key in results_epoch.keys():
            results_epoch[key] /= len(trainloader)
            results_epoch[key] = results_epoch[key].item()
    return results_epoch


def test_loop(fabric, net, testloader):
    """Evaluate the network on the entire test set."""
    # Alice: fabric is not used
    net.eval()

    with torch.no_grad():
        results_epoch = {
            key: torch.tensor(0.0, device=net.device)
            for key in net.signature.__required_keys__
        }
        for batch_idx, batch in track(
            enumerate(testloader),
            total=len(testloader),
            description="Validating...",
        ):
            results = net.validation_step(batch, batch_idx)

            for key in results_epoch.keys():
                value = results[key]

                # hardening code --- begin ---
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, device=net.device)

                if value.shape != results_epoch[key].shape:
                    value = value.reshape(results_epoch[key].shape)
                # hardening code --- end ---

                results_epoch[key] += value

    for key in results_epoch.keys():
        results_epoch[key] /= len(testloader)
        results_epoch[key] = results_epoch[key].item()
    return results_epoch
