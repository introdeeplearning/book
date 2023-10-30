import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import copy

# Set device as GPU if available or CPU otherwise
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Fix a random seed
torch.manual_seed(0)

# Load the MNIST training and test datasets
mnist_train = datasets.MNIST(
    "./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
mnist_test = datasets.MNIST(
    "./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = data.DataLoader(
    mnist_train, batch_size=64, shuffle=True
)
test_loader = data.DataLoader(
    mnist_test, batch_size=64, shuffle=False
)

# Define a neural network
net = nn.Sequential(  # input shape (N, 1, 28, 28)
    nn.Conv2d(1, 5, 5),  # (N, 5, 24, 24)
    nn.ReLU(),
    nn.Conv2d(5, 5, 3),  # (N, 5, 22, 22)
    nn.ReLU(),
    nn.Conv2d(5, 3, 3),  # (N, 3, 20, 20)
    nn.ReLU(),
    nn.Flatten(),  # (N, 3 * 16 * 16) = (N, 1200)
    nn.Linear(1200, 128),  # (N, 128)
    nn.ReLU(),
    nn.Linear(128, 10),  # output shape (N, 10)
).to(device)

# Save the initial state of the neural network
initial_state = copy.deepcopy(net.state_dict())

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the optimizers that we want to compare. Each entry in the
# list is a tuple of a label (for the plot) and an optimizer
optimizers = [
    # For SGD we use a learning rate of 0.001
    (
        "SGD",
        optim.SGD(net.parameters(), lr=1e-3),
    ),
    (
        "SGD with momentum",
        optim.SGD(net.parameters(), lr=1e-3, momentum=0.9),
    ),
    (
        "Nesterov SGD",
        optim.SGD(
            net.parameters(), lr=1e-3, momentum=0.9, nesterov=True
        ),
    ),
    # For the adaptive optimization methods we use the default
    # hyperparameters
    (
        "RMSprop",
        optim.RMSprop(net.parameters()),
    ),
    (
        "Adagrad",
        optim.Adagrad(net.parameters()),
    ),
    (
        "Adadelta",
        optim.Adadelta(net.parameters()),
    ),
    (
        "Adam",
        optim.Adam(net.parameters()),
    ),
]

def compute_test_loss_and_accuracy():
    total_test_loss = 0.0
    correct_count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = net(images)
            loss = loss_fn(output, labels)

            total_test_loss += loss.item() * images.size(0)
            pred_labels = torch.max(output, dim=1).indices
            correct_count += torch.sum(
                pred_labels == labels
            ).item()

    avg_test_loss = total_test_loss / len(mnist_test)
    accuracy = correct_count / len(mnist_test)

    return (avg_test_loss, accuracy)


loss_plots = []
accuracy_plots = []

test_interval = 100

for _, optimizer in optimizers:
    train_losses = []
    accuracies = []
    print(optimizer)

    with torch.no_grad():
        net.load_state_dict(initial_state)

    i = 0
    for e in range(5):
        print(f"Epoch {e+1}")
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (i + 1) % test_interval == 0:
                (
                    test_loss,
                    accuracy,
                ) = compute_test_loss_and_accuracy()
                print(accuracy)
                accuracies.append(accuracy)

            i += 1

    loss_plots.append(train_losses)
    accuracy_plots.append(accuracies)

WINDOW = 200

_, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax1.set_yscale("log")
ax2.set_yscale("logit")
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_formatter(NullFormatter())
for (label, _), train_losses, accuracies in zip(
    optimizers, loss_plots, accuracy_plots
):
    ax1.plot(
        [
            sum(train_losses[max(0,i-WINDOW) : i]) / min(i, WINDOW)
            for i in range(1,len(train_losses))
        ],
        label=label,
    )
    ax2.plot(
        range(0, len(accuracies) * test_interval, test_interval),
        accuracies,
        label=label,
    )

ax1.legend()

plt.tight_layout()
plt.savefig("../plots/mnist_optim.pdf", bbox_inches="tight")
