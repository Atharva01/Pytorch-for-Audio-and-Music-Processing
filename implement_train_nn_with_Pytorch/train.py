import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 1. download dataset
# 2. create dataloader
# 3. build model
# 4. train model
# 5. save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions,targets)
        
        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    print(f"Loss:{loss.item()}")
    
    
def train(model, data_loader, loss_fn, optimiser, device,epochs):
    for i in range(epochs):
        print(f"Epoch:{i+1}")
        train_one_epoch(model,data_loader, loss_fn, optimiser, device)
        print(f"-----------------------------------------------------")
    print("Training Completed")
    
def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data


if __name__ =="__main__":
    train_data,_ = download_mnist_datasets()
    print(f"MNIST Dataset downloaded")
    
    #create a dataloader
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    
    #build a model
    feed_forward_net = FeedForwardNet().to(device)
    
    
    #instantiate loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),lr=LEARNING_RATE)
    
    #train a model
    print(f"Training on {device}")
    train(feed_forward_net, train_data_loader,loss_fn, optimiser,device,EPOCHS) 
    
    #save the model
    
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print(f"Trained and Stored at feedforwardnet.pth")
