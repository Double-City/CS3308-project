import os
import numpy as np
from dataset import get_data, normalize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )

        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == '__main__':
    ######################## Get train dataset ########################
    X_train = get_data('dataset')
    X_train = torch.from_numpy(normalize(X_train))
    ####################################################################
    ######################## Implement you code here ###################
    ####################################################################
    
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    z_dim = 64
    model = VAE(z_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 300
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data[0])
        
            loss = loss_function(recon_batch, data[0], mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(dataloader.dataset):.6f}')


    def linear_interpolation(model, z1, z2, alpha_values):
        generated_images = []
        for alpha in alpha_values:
            z = alpha * z1 + (1 - alpha) * z2
        
            z = z.unsqueeze(0)
            with torch.no_grad():
                generated_image = model.decode(z).squeeze(0)
            generated_images.append(generated_image)
        return generated_images

    alpha_values = np.linspace(0, 1, num=5)
    
    fig, axs = plt.subplots(15, 5, figsize=(24, 18))
    
    
    for j in range(15):
        z1 = torch.randn(z_dim)
        z2 = torch.randn(z_dim)
        generated_images = linear_interpolation(model, z1, z2, alpha_values)
        for i, image in enumerate(generated_images):
            axs[j-1, i].imshow(image.permute(1, 2, 0))
            axs[j-1, i].axis('off')
    plt.show()