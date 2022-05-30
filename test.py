from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, classes):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    actual = []
    predicted = []

    with torch.no_grad():
        for X, y in dataloader:
            
            X, y = X.to(device), y.to(device)
            pred = model(X)

            actual.append(y[0])
            predicted.append(pred[0].argmax(0))

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # print("test0 :",classes[pred[0].argmax(0)])
            # print("test1 :",classes[y[0]], "\n")
            
            # print(actual)
            # print(predicted)

    test_loss /= num_batches
    correct /= size
    return 100*correct, test_loss, f1_score(actual, predicted,average="macro")

def train_epochs(epochs, save_name,classes):
    writer = SummaryWriter()
    for t in range(epochs):
        print(f'Epoch [{t:>3d}/{epochs:>3d}]')
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy, loss, f1_score = test(test_dataloader, model, loss_fn,classes)
        
        writer.add_scalar("Accuracy/train",accuracy, t+1)
        writer.add_scalar("Loss/train", loss, t+1)
        writer.add_scalar("F1-score",f1_score,t+1)
        print("f1 :",f1_score)
        #print(f"Test Error:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")

    writer.flush()
    writer.close()
    print("Done!")
    torch.save(model.state_dict(), save_name)
    print(f"Saved PyTorch Model State to {save_name}")
        

def test_model(model_name,classes):

    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_name))    
    model.eval()
    count = 0

    for i in range(len(test_data)):
        with torch.no_grad():
            x, y = test_data[i][0], test_data[i][1]
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            #print(pred[0].argmax(0))
            #print(f'Predicted: "{predicted}", Actual: "{actual}"')
            if predicted == actual:
                count += 1
    print(f'Got {count} good prediction over {len(test_data)} ({count/len(test_data)*100}% accuracy)')

classes = [
"T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot",
]

train_epochs(250, "model.pth",classes)
test_model("model.pth", classes)