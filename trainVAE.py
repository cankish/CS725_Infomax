from torch.optim import Adam
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import VAE
from utils import get_loaders

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def plot_reconstruction(img, recons):
    """
    Plot the original and reconstructed images during training
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))

    for j in range(5):
        axes[0][j].imshow(np.squeeze(img[j].detach().cpu().numpy()), cmap='gray')
        axes[0][j].axis('off')

    for j in range(5):
        axes[1][j].imshow(np.squeeze(recons[j].detach().cpu().numpy()), cmap='gray')
        axes[1][j].axis('off')

    plt.tight_layout(pad=0.)
    plt.show()

def show_tensor_images(image_tensor, num_images=25, size=(3, 32, 32), nrow=5, show=True, gen = False, save = False):

    # if gen:
    #   image_tensor = (image_tensor + 1)/2

    # print(image_tensor.max(), image_tensor.min())
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    # plt.imshow(torch.tensor( image_grid.permute(1, 2, 0).squeeze() , dtype = torch.uint8 ) )
    plt.imshow(torch.tensor( image_grid.permute(1, 2, 0).squeeze() ))

    if show:
        plt.show()
    if save:
      for i in range(image_tensor.shape[0]):
        save_image(fake[i], gen_path + f"{i}.png")

print("Start training VAE...")
BCE_loss = nn.BCELoss()
model = VAE().to(device)

from prettytable import PrettyTable
def count_parameters(model):
  table = PrettyTable(["Modules", "Parameters"])
  total_params = 0
  for name, parameter in model.named_parameters():
      if not parameter.requires_grad: continue
      param = parameter.numel()
      table.add_row([name, param])
      total_params+=param
  print(table)

  print(f"Total Trainable Params: {total_params}")
  return total_params

count_parameters(model.Encoder)
count_parameters(model.Decoder)

model.train()
epochs = 50
batch_size = 128
lr = 1e-3

loaders = get_loaders('cifar10', batch_size=batch_size)
train_loader = loaders['train']
optimizer = Adam(model.parameters(), lr=lr)


for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        # x = x.view(batch_size, x_dim)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

        # plt.imshow(torch.permute(x[0], (1,2,0)).detach().cpu().numpy())
        # plt.imshow(torch.permute(x_hat[0], (1,2,0)).detach().cpu().numpy())
        # plt.show()
    show_tensor_images(x, gen = True)
    show_tensor_images(x_hat, gen = True)
    # plot_reconstruction(x, x_hat)
    # if (epoch+1)%5==0:
    #     plot_reconstruction(x, x_hat)


    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))

    save_dir = f'vae_{epoch+1}.pth'
    torch.save(model.state_dict(), save_dir)

print("Finish!!")