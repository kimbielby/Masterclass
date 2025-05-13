from Utils import *

train_accuracy = []
train_loss = []

def train(model, train_loader, num_samples_batch, device, lr = 1e-4):
    accuracy = 0.0
    loss = 0.0

    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        output = model(inputs)

        optimiser = get_optimiser()
        optimiser = optimiser(model.parameters(), lr=lr)
        optimiser.zero_grad()

        loss_function = get_loss_function()
        batch_loss = loss_function(output, targets)
        batch_loss.backward()

        optimiser.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output=output, target=targets, N=num_samples_batch)

    train_accuracy.append(accuracy)
    train_loss.append(loss)
    print(f"Train Loss: {loss:.3f}, Train Accuracy: {accuracy * 100:.3f}")

""" Getters for Lists """
def get_train_accuracy_list():
    return train_accuracy

def get_train_loss_list():
    return train_loss
