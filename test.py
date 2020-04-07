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

        # 모델의 레이어들 생성
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 10)

        # 순전파 과정
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
    
    # 데이터셋 디렉토리 생성
    os.makedirs("../mnist_data", exist_ok=True)

    # 학습 데이터셋 생성. 이미지: 0.1307, 라벨: 0.3081로 Normalize시킨다.
    trn_dataset = datasets.MNIST('../mnist_data/',
                             download=True,
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),  # image to Tensor
                                 transforms.Normalize((0.1307,), (0.3081,))  # image, label
                             ]))
    
    # 검증 데이터셋 생성. 이미지: 0.1307, 라벨: 0.3081로 Normalize시킨다.
    val_dataset = datasets.MNIST("../mnist_data/",
                             download=False,
                             train=False,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307, ),(0.3081, ))
                           ]))

    # pytorch 데이터 로드 생성
    trn_loader = Data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # cuda 사용 가능 확인
    use_cuda = torch.cuda.is_available()
    # setting device on GPU if available, else CPU
    
    # cuda 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    
    # cuda 장치의 이름 출력
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # MLP 모델 생성
    mlp = MLP()
    
    # cuda 장치로 변환
    if use_cuda:
        mlp = mlp.cuda()

    # loss 함수 생성
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3
    
    # optimizer 생성
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    num_batches = len(trn_loader)

    
    # 학습 과정
    trn_loss_list = []
    val_loss_list = []
    for epoch in range(epoches):
        trn_loss = 0.0
        for i, data in enumerate(trn_loader):
                       
            x, label = data
            if use_cuda:
                x = x.cuda()
                label = label.cuda()

            # optimizer gradient 초기화
            optimizer.zero_grad()
            
            # model 결과
            model_output = mlp(x)
                        
            # loss 계산
            loss = criterion(model_output, label)

            # gradient 계산
            loss.backward()
            
            # optimizer를 통해 update
            optimizer.step()

            # loss 출력
            trn_loss += loss.item()
            del loss
            del model_output
            
            # 100 번째 iteration 마다 검증
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


