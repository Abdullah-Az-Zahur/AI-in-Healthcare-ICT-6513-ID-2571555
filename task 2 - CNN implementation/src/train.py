def train(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    if device is None:
        device = next(model.parameters()).device

    for x, y in loader:
<<<<<<< HEAD
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = loss_fn(out, y)
=======
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = loss_fn(output, y)
>>>>>>> efc43190975a9e3d4d6764e561cef8ba43eb8cef

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)