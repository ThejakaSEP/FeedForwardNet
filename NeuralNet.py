#This is a Feed forward Network

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1. download dataset
# 2.create data dataloader - a class to fetch data in batches. This is iterable.
# 3.build model
# 4.train
# 5.save trained model
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class FeedForwardNet(nn.Module):
    #Constructor
    def __init__(self):
        super().__init__() #Invoke the constructor of Base class(Module class)
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

        self.softmax = nn.Softmax(dim=1) #Basic transformation - Sort of a normalization

    def forward(self,input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download = True,
        train = True,
        transform = ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data,validation_data

def train_one_epoch(model,data_loader,loss_fn,optimiser,device):
    for inputs,targets in data_loader:
        inputs,targets = inputs.to(device),targets.to(device)

        #1. Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions,targets)

        #2. backpropagate loss and update weights
        optimiser.zero_grad()#resetti ng the gradient before the next batch
        loss.backward() #Apply back propagation
        optimiser.step() #Update the weights

    print(f"Loss: {loss.item()}")
    # print("Hiiiii")

def test(data_loader,device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)


def train(model,data_loader,loss_fn,optimiser,device,epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model,data_loader,loss_fn,optimiser,device)
        print("----------------")
    print("Training is done")

if __name__ == "__main__":
    #download MNIST dataset
    train_data,_ = download_mnist_datasets()
    # print(train_data)

    #2.Create a data loader
    train_data_loader = DataLoader(train_data,batch_size=BATCH_SIZE)
    # print(train_data_loader)

    #3.Build the model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)

    #instantiate loss function + optimiser

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),lr=LEARNING_RATE)

    # Train model
    train(feed_forward_net,train_data_loader,loss_fn,optimiser,device,EPOCHS)


    #5. Save the model
    torch.save(feed_forward_net.state_dict(),"feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")

    # print(train_data)