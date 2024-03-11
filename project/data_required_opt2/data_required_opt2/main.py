import os
import numpy as np
from dataset import get_data, normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

import mypca
from mypca import pca_manual

import mytsne
from mytsne import TSNE

BATCH_SIZE = 64

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16* 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, x.view(x.size(0), -1)
    
    def get_mid(self, x, str):
        x = self.conv1(x)
        if str == "conv1" :
            return x
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        if str == "conv2" :
            return x
        x = self.pool2(F.relu(x))
        x = x.view(-1, 16* 5 * 5)
        x = F.relu(self.fc1(x))
        if str == "fc1":
            return x
        x = F.relu(self.fc2(x))
        if str == "fc2":
            return x
        x = self.fc3(x)
        if str == "final":
            return x
    
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x, x.view(x.size(0), -1)
    
    def get_mid(self, x, str):
        x = self.conv1(x)
        if str == "conv1":
            return x
        x = F.relu(self.bn1(x))
        if str == "bn1":
            return x
        x = self.pool1(x)
        
        x = self.conv2(x)
        if str == "conv2":
            return x
        x = F.relu(self.bn2(x))
        if str == "bn2":
            return x
        x = self.pool2(x)
        
        x = self.conv3(x)
        if str == "conv3":
            return x
        x = F.relu(self.bn3(x))
        if str == "bn3":
            return x
        x = x.view(-1, 64 * 8 * 8)
        
        x = self.fc1(x)
        if str == "fc1":
            return x
        x = F.relu(self.bn4(x))
        if str == "bn4":
            return x
        x = self.fc2(x)
        if str == "final":
            return x

if __name__ == '__main__':
    ######################## Get train/test dataset ########################
    X_train, X_test, Y_train, Y_test = get_data('dataset')
    X_train = torch.from_numpy(normalize(X_train))
    Y_train = torch.from_numpy(Y_train).long()
    X_test = torch.from_numpy(normalize(X_test))
    Y_test = torch.from_numpy(Y_test).long()
    train_dataset = Data.TensorDataset(X_train, Y_train)
    test_dataset = Data.TensorDataset(X_test, Y_test)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    ######################## Define Net ########################
    net = LeNet()
    # net = MyNet()

    ######################## Define loss and optimizer ########################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0008, momentum=0.9)

    ######################## Train the network ########################
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    num_epoch = 360
    
    for epoch in range(num_epoch):
        running_loss = 0.0
        running_corrects = 0
        net.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            my_inputs, my_labels = data
            
            optimizer.zero_grad()
            
            outputs, features = net(inputs.float()) 

            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects.double() / len(train_dataset)

            

        test_loss, test_accuracy = 0, 0
        net.eval()
    

        with torch.no_grad():
            test_corrects = 0
            for data in test_loader:
                images, labels = data
                outputs, features = net(images.float()) 
            
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_corrects += torch.sum(predicted == labels.data)
           
            test_loss /= len(test_loader)
            test_accuracy = test_corrects.double() / len(test_dataset)
            
        print(f'Epoch {epoch + 1}/{num_epoch}: 'f'Train Loss: {train_loss:.6f} Train Acc: {train_accuracy:.4f} 'f'Test Loss: {test_loss:.6f} Test Acc: {test_accuracy:.4f}')        

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    # features_conv = lenet.get_mid(X_test.float(), "conv2")
    # features_conv = features_conv.numpy()
    # print(features_conv)
    # 绘制训练损失和测试损失随 epoch 变化的曲线图
    plt.figure()
    plt.plot(range(1, 1 + num_epoch), train_losses, label="Train Loss")
    plt.plot(range(1, 1 + num_epoch), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()

    # # 绘制训练准确率和测试准确率随 epoch 变化的曲线图
    plt.figure()
    plt.plot(range(1, 1 + num_epoch), train_accuracies, label="Train Acc")
    plt.plot(range(1, 1 + num_epoch), test_accuracies, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()

    ######################## Visualize middle layer outputs ########################
    num_samples = 888
    X_test = X_test[:num_samples]
    Y_test = Y_test[:num_samples]

    with torch.no_grad():
        outputs, _ = net(X_test.float()) 
        features_conv = net.get_mid(X_test.float(), "conv2")
        features_fc = net.get_mid(X_test.float(), "fc1")
        features_final = net.get_mid(X_test.float(), "final")
        
    features_conv = features_conv.view(features_conv.shape[0], -1).detach().numpy()
    features_fc = features_fc.view(features_fc.shape[0], -1).detach().numpy()
    features_final = features_final.view(features_final.shape[0], -1).detach().numpy()

    # 使用PCA进行降维
    #pca = PCA(n_components=2)
    #pca_result = pca.fit_transform(features)
    pca_result_conv = pca_manual(features_conv, n_components=2)
    pca_result_fc = pca_manual(features_fc, n_components=2)
    pca_result_final = pca_manual(features_final, n_components=2)

    
    # 使用tSNE进行降维
    #tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0)
    tsne_manual = TSNE()
    tsne_result_conv = tsne_manual.fit_transform(features_conv)
    tsne_result_fc = tsne_manual.fit_transform(features_fc)
    tsne_result_final = tsne_manual.fit_transform(features_final)


    # 可视化PCA降维结果
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(27, 8))
    for label in range(10):
        ix = np.where(Y_test == label)
        axs[0].scatter(pca_result_conv[ix, 0], pca_result_conv[ix, 1], label=str(label))
    axs[0].legend()
    axs[0].set_xlabel('PCA Component 1')
    axs[0].set_ylabel('PCA Component 2')
    axs[0].set_title('Visualization of MyNet Conv Layer Outputs Using PCA_manual')

    # 可视化t-SNE降维结果
    for label in range(10):
        ix = np.where(Y_test == label)
        axs[1].scatter(pca_result_fc[ix, 0], pca_result_fc[ix, 1], label=str(label))
    axs[1].legend()
    axs[1].set_xlabel('PCA Component 1')
    axs[1].set_ylabel('PCA Component 2')
    axs[1].set_title('Visualization of MyNet FC Layer Outputs Using PCA_manual')
    
    for label in range(10):
        ix = np.where(Y_test == label)
        axs[2].scatter(pca_result_final[ix, 0], pca_result_final[ix, 1], label=str(label))
    axs[2].legend()
    axs[2].set_xlabel('PCA Component 1')
    axs[2].set_ylabel('PCA Component 2')
    axs[2].set_title('Visualization of MyNet Final Layer Outputs Using PCA_manual')


    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(27, 8))
    for label in range(10):
        ix = np.where(Y_test == label)
        axs[0].scatter(tsne_result_conv[ix, 0], tsne_result_conv[ix, 1], label=str(label))
    axs[0].legend()
    axs[0].set_xlabel('t-SNE Component 1')
    axs[0].set_ylabel('t-SNE Component 2')
    axs[0].set_title('Visualization of MyNet Conv Layer Outputs Using t-SNE_manual')
    
    for label in range(10):
        ix = np.where(Y_test == label)
        axs[1].scatter(tsne_result_fc[ix, 0], tsne_result_fc[ix, 1], label=str(label))
    axs[1].legend()
    axs[1].set_xlabel('t-SNE Component 1')
    axs[1].set_ylabel('t-SNE Component 2')
    axs[1].set_title('Visualization of MyNet FC Layer Outputs Using t-SNE_manual')
    
    for label in range(10):
        ix = np.where(Y_test == label)
        axs[2].scatter(tsne_result_final[ix, 0], tsne_result_final[ix, 1], label=str(label))
    axs[2].legend()
    axs[2].set_xlabel('t-SNE Component 1')
    axs[2].set_ylabel('t-SNE Component 2')
    axs[2].set_title('Visualization of MyNet Final Layer Outputs Using t-SNE_manual')
    
    plt.show()
    
    
    