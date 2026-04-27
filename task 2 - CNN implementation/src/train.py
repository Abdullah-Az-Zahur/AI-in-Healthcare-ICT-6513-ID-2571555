def train(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0

    for x, y in loader:
        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
