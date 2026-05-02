import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# here is a implementation that's not the best for use in projects.. but the better to understand fully



# HyperParametres:

NUM_INPUTS = 784
NUM_OUTPUTS = 10
NUM_HIDDEN_LAYERS = 256
LR = 0.1
BATCH_SIZE = 256
NUM_EPOCHS = 10
SIGMA = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")


transform = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download = True
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

Labels = ["t-shirt","pantalon","pull","robe","manteau",
          "sandale","chemise","basket","sac","bottine"]

def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    preds = y_hat.argmax(dim=1)
    return (preds == y).float().mean().item()

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            total_loss += loss_fn(y_hat, y)
            total_acc += accuracy(y_hat, y) * len(y)
            n += len(y)
    return total_loss / n, total_acc / n

def train(model, train_loader, test_loader, loss_fn, optimizer, num_epochs):
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_acc += accuracy(y_hat, y) * len(y)
            n += len(y)

        train_loss = total_loss / n
        train_acc = total_acc / n
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
              f"test loss {test_loss:.4f}, acc {test_acc:.3f}")

    return history


def plot_history(history, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["test_acc"], label="test")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=120)
    plt.show()



class MLPScratch(nn.Module):

    def __init__(self, num_inputs, num_hiddens, num_outputs, sigma = 0.01):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        X = X.reshape(-1, self.W1.shape[0])
        H = torch.relu(X @ self.W1 + self.b1)
        O = H @ self.W2 + self.b2
        return O

model_scratch = MLPScratch(NUM_INPUTS, NUM_HIDDEN_LAYERS, NUM_OUTPUTS, SIGMA).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_scratch.parameters(), lr=LR)
history_scratch = train(model_scratch, train_loader, test_loader, loss_fn, optimizer, NUM_EPOCHS)
plot_history(history_scratch)