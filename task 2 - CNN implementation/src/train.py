def train(model, loader, loss_fn, optimizer, device=None):
    model.train()
    total_loss = 0

    if device is None:
        device = next(model.parameters()).device

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
