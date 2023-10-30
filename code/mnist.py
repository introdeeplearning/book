import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

# We use the GPU if available. Otherwise, we use the CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# We fix a random seed. This is not necessary for training a
# neural network, but we use it here to ensure that the same
# plot is created on every run.
torch.manual_seed(0)

# The torch.utils.data.Dataset class is an abstraction for a
# collection of instances that has a length and can be indexed
# (usually by integers).
# The torchvision.datasets module contains functions for loading
# popular machine learning datasets, possibly downloading and
# transforming the data.

# Here we load the MNIST dataset, containing 28x28 grayscale images
# of handwritten digits with corresponding labels in
# {0, 1, ..., 9}.

# First load the training portion of the data set, downloading it
# from an online source to the local folder ./data (if it is not
# yet there) and transforming the data to PyTorch Tensors.
mnist_train = datasets.MNIST(
    "./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
# Next load the test portion
mnist_test = datasets.MNIST(
    "./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

# The data.utils.DataLoader class allows iterating datasets for
# training and validation. It supports, e.g., batching and 
# shuffling of datasets.

# Construct a DataLoader that when iterating returns minibatches
# of 64 instances drawn from a random permutation of the training
# dataset
train_loader = data.DataLoader(
    mnist_train, batch_size=64, shuffle=True
)
# The loader for the test dataset does not need shuffling
test_loader = data.DataLoader(
    mnist_test, batch_size=64, shuffle=False
)

# Define a neural network with 3 convolutional layers, each
# followed by a ReLU activation and then two affine layers,
# the first followed by a ReLU activation
net = nn.Sequential(  # input shape (N, 1, 28, 28)
    nn.Conv2d(1, 5, 5),  # (N, 5, 24, 24)
    nn.ReLU(),
    nn.Conv2d(5, 5, 5),  # (N, 5, 20, 20)
    nn.ReLU(),
    nn.Conv2d(5, 3, 5),  # (N, 3, 16, 16)
    nn.ReLU(),
    nn.Flatten(),  # (N, 3 * 16 * 16) = (N, 768)
    nn.Linear(768, 128),  # (N, 128)
    nn.ReLU(),
    nn.Linear(128, 10),  # output shape (N, 10)
).to(device)

# Define the loss function. For every natural number d, for
# e_1, e_2, ..., e_d the standard basis vectors in R^d, for L the
# d-dimensional cross-entropy loss function, and for A the
# d-dimensional softmax activation function, the function loss_fn
# defined here satisfies for all x in R^d and all natural numbers
# i in [0,d) that
# loss_fn(x, i) = L(A(x), e_i).
# The function loss_fn also accepts batches of inputs, in which
# case it will return the mean of the corresponding outputs.
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer. We use the Adam SGD optimization method.
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# This function computes the average loss of the model over the
# entire test set and the accuracy of the model's predictions.
def compute_test_loss_and_accuracy():
    total_test_loss = 0.0
    correct_count = 0
    with torch.no_grad():
        # On each iteration the test_loader will yield a 
        # minibatch of images with corresponding labels
        for images, labels in test_loader:
            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)
            # Compute the output of the neural network on the 
            # current minibatch
            output = net(images)
            # Compute the mean of the cross-entropy losses
            loss = loss_fn(output, labels)
            # For the cumulative total_test_loss, we multiply loss
            # with the batch size (usually 64, as specified above,
            # but might be less for the final batch).
            total_test_loss += loss.item() * images.size(0)
            # For each input, the predicted label is the index of 
            # the maximal component in the output vector.
            pred_labels = torch.max(output, dim=1).indices
            # pred_labels == labels compares the two vectors
            # componentwise and returns a vector of booleans. 
            # Summing over this vector counts the number of True 
            # entries.
            correct_count += torch.sum(
                pred_labels == labels
            ).item()
    avg_test_loss = total_test_loss / len(mnist_test)
    accuracy = correct_count / len(mnist_test)
    return (avg_test_loss, accuracy)


# Initialize a list that holds the computed loss on every
# batch during training
train_losses = []

# Every 10 batches, we will compute the loss on the entire test
# set as well as the accuracy of the model's predictions on the
# entire test set. We do this for the purpose of illustrating in 
# the produced plot the generalization capability of the ANN. 
# Computing these losses and accuracies so frequently with such a 
# relatively large set of datapoints (compared to the training 
# set) is extremely computationally expensive, however (most of 
# the training runtime will be spent computing these values) and 
# so is not advisable during normal neural network training.
# Usually, the test set is only used at the very end to judge the
# performance of the final trained network. Often, a third set of
# datapoints, called the validation set (not used to train the 
# network directly nor to evaluate it at the end) is used to 
# judge overfitting or to tune hyperparameters.
test_interval = 10
test_losses = []
accuracies = []

# We run the training for 5 epochs, i.e., 5 full iterations
# through the training set.
i = 0
for e in range(5):
    for images, labels in train_loader:
        # Move the data to the device
        images = images.to(device)
        labels = labels.to(device)

        # Zero out the gradients
        optimizer.zero_grad()
        # Compute the output of the neural network on the current
        # minibatch
        output = net(images)
        # Compute the cross entropy loss
        loss = loss_fn(output, labels)
        # Compute the gradients
        loss.backward()
        # Update the parameters of the neural network
        optimizer.step()

        # Append the current loss to the list of training losses.
        # Note that tracking the training loss comes at 
        # essentially no computational cost (since we have to 
        # compute these values anyway) and so is typically done 
        # during neural network training to gauge the training 
        # progress.
        train_losses.append(loss.item())

        if (i + 1) % test_interval == 0:
            # Compute the average loss on the test set and the
            # accuracy of the model and add the values to the
            # corresponding list
            test_loss, accuracy = compute_test_loss_and_accuracy()
            test_losses.append(test_loss)
            accuracies.append(accuracy)

        i += 1

fig, ax1 = plt.subplots(figsize=(12, 8))
# We plot the training losses, test losses, and accuracies in the
# same plot, but using two different y-axes
ax2 = ax1.twinx()

# Use a logarithmic scale for the losses
ax1.set_yscale("log")
# Use a logit scale for the accuracies
ax2.set_yscale("logit")
ax2.set_ylim((0.3, 0.99))
N = len(test_losses) * test_interval
ax2.set_xlim((0, N))
# Plot the training losses
(training_loss_line,) = ax1.plot(
    train_losses,
    label="Training loss (left axis)",
)
# Plot test losses
(test_loss_line,) = ax1.plot(
    range(0, N, test_interval),
    test_losses,
    label="Test loss (left axis)",
)
# Plot the accuracies
(accuracies_line,) = ax2.plot(
    range(0, N, test_interval),
    accuracies,
    label="Accuracy (right axis)",
    color="red",
)
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_formatter(NullFormatter())

# Put all the labels in a common legend
lines = [training_loss_line, test_loss_line, accuracies_line]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels)

plt.tight_layout()
plt.savefig("../plots/mnist.pdf", bbox_inches="tight")
