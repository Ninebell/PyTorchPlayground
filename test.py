import os
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(28*28, 5050)
        self.fc2 = nn.Linear(5050, 5050)
        self.fc3 = nn.Linear(5050, 5050)
        self.fc4 = nn.Linear(5050, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=1)
        return x


if __name__ == "__main__":

    batch_size=64

    epoches = 10

    os.makedirs("../mnist_data", exist_ok=True)

    trn_dataset = datasets.MNIST('../mnist_data/',
                             download=True,
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),  # image to Tensor
                                 transforms.Normalize((0.1307,), (0.3081,))  # image, label
                             ]))
    val_dataset = datasets.MNIST("../mnist_data/",
                             download=False,
                             train=False,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307, ),(0.3081, ))
                           ]))

    trn_loader = Data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    use_cuda = torch.cuda.is_available()
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    mlp = MLP()
    if use_cuda:
        mlp = mlp.cuda()

    criterion = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    num_batches = len(trn_loader)

    trn_loss_list = []
    val_loss_list = []
    for epoch in range(epoches):
        trn_loss = 0.0
        for i, data in enumerate(trn_loader):
            x, label = data
            if use_cuda:
                x = x.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            model_output = mlp(x)

            loss = criterion(model_output, label)

            loss.backward()
            optimizer.step()

            trn_loss += loss.item()
            del loss
            del model_output
            if (i + 1) % 100 == 0:  # every 100 mini-batches
                with torch.no_grad():  # very very very very important!!!
                    val_loss = 0.0
                    for j, val in enumerate(val_loader):
                        val_x, val_label = val
                        if use_cuda:
                            val_x = val_x.cuda()
                            val_label = val_label.cuda()
                        val_output = mlp(val_x)
                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss

                print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(
                    epoch + 1, epoches, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader)
                ))

                trn_loss_list.append(trn_loss / 100)
                val_loss_list.append(val_loss / len(val_loader))
                trn_loss = 0.0


