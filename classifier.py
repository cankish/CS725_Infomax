from collections import OrderedDict
from pathlib import Path
import copy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import get_features
from trainer import get_models, load_checkpoint
import numpy as np
from models import VAE


def test_model(model, test_loader):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    criterion = nn.CrossEntropyLoss()
    correct = 0
    l1 = []
    l2 = []
    batch_size = 128
    model.eval()

    for data, target in test_loader:
        if data.shape[0]!=batch_size:
            break
        data, target = data.to(device), target.to(device)
        # y, M = encoder(data)
        output = model(data).to(device)

        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        # correct += (pred == target).sum().cpu()
        l1.extend(target.data.cpu())
        l2.extend(pred.data.cpu())
        for i in range(len(target.data)):
            # print(i, target.data[i])
            l = target.data[i]
            class_correct[l] += correct[i].item()
            class_total[l] += 1
            # precision[l] =
            # recall[l] =

    test_loss = test_loss/len(test_loader.dataset)
    print(f'Test Loss: {test_loss}')
    # print('Acc',correct/len(test_loader.dataset))
    # micro = get_microF1(np.asarray(l1),np.asarray(l2))
    # macro = get_macroF1(np.asarray(l1),np.asarray(l2))
    for i in range(10):
        if class_total[i] > 0:
            print(f'Test Accuracy of {str(i), 100 * class_correct[i] / class_total[i]}: ({np.sum(class_correct[i])}/{np.sum(class_total[i])})')
        else:
            print(f'Test Accuracy of {(classes[i])}: N/A (no training examples)')

    t = 100. * np.sum(class_correct) / np.sum(class_total)
    print(f'Test Accuracy (Overall): {t} {np.sum(class_correct)}/{np.sum(class_total)}')

    return

def train_model(
        model,
        loaders,
        optimizer,
        scheduler,
        num_epochs,
        device=torch.device('cpu')):

    criterion = nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float('-inf')

    for epoch in range(1,num_epochs+1):
        print(f'\nEpoch {epoch}/{num_epochs}\n' + '-' * 10)

        # Each epoch has a training and validation phase
        for phase, loader in loaders.items():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                correct = torch.sum(preds == labels.data)
                running_corrects += correct
                if i%10 == 0 and False:
                    print(f'\tIter {i} Loss: {loss}, Acc: {correct/len(labels)}')

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            # test_model(model, loaders['valid'])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, ', end='')
            print(f'LR: {scheduler.get_lr()[0]:.6f}')

            # deep copy the model
            if phase != 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def eval_encoder(
        encoder,
        loaders,
        eval_model,
        optimizer,
        scheduler,
        train_epochs=5,
        batch_size=128,
        model='dim',
        feature_layer='y',
        device=torch.device('cpu')):

    """
    Args:
        encoder (nn.Module): Encoder function to generate features from,
            with `forward` function that returns y, M
        loaders (dict(DataLoader)): dictionary with keys 'valid' and 'train',
            which contain DataLoaders
        eval_model (nn.Module): Module to train on encoder features
        train_epochs (int): Number of training epochs to train eval_model
        batch_size  (int): Batch size for training eval_model
        feature_layer (str): One of 'y', 'fc', 'conv' (case insensitive). The
            layer in the encoder from which to generate features
        device (torch.device): 'cpu' or 'cuda'
    Returns
        (int): Accuracy of the eval_model trained on encoder features
    """
    encoder.eval()
    if(model == 'dim'):

      feature_layer = feature_layer.lower()
      if feature_layer == 'y':
          forward_fn = lambda images: encoder(images)[0]
      elif feature_layer == 'fc':
          forward_fn = lambda images: encoder.f1(encoder.f0(encoder.C(images)))
      elif feature_layer == 'conv':
          # Encoder needs to implement an interface
          forward_fn = lambda images: encoder.f0(encoder.C(images))
      else:
          raise ValueError(f"feature_layer was {feature_layer}"
                  + "but should be one of 'y', 'fc', 'conv'")

    if(model == 'vae'):
      forward_fn = lambda images: encoder(images)[2]


    # Train classifier on encoder features
    train_features, train_labels = get_features(forward_fn, loaders['train'], device)
    eval_features, eval_labels = get_features(forward_fn, loaders['valid'], device)
    encoder.to('cpu')

    train_set = TensorDataset(train_features, train_labels)
    eval_set = TensorDataset(eval_features, eval_labels)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    loaders = OrderedDict(train=train_loader, valid=eval_loader)

    eval_model = train_model(eval_model, loaders, optimizer, scheduler, train_epochs, device)
    save_dir = 'checkpoints/classifier.pth'
    torch.save(eval_model.state_dict(), save_dir)



if __name__ == "__main__":
    from models import Encoder, Classifier
    from utils import get_loaders
    from torch.optim.lr_scheduler import StepLR
    import argparse

    parser = argparse.ArgumentParser(description='Deep Info Max PyTorch')
    parser.add_argument('--batch_size', type=int, default=32+16,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('-a', '--alpha', type=float, default=1,
                        help='Coefficient for global DIM (default: 1)')
    parser.add_argument('-b', '--beta', type=float, default=0,
                        help='Coefficient for local DIM (default: 0)')
    parser.add_argument('-g', '--gamma', type=float, default=1,
                        help='Coefficient prior matching (default: 1)')

    parser.add_argument('-model', '--model', type=str, default='dim',
                        help='Model used for classification features')


    args = parser.parse_args()
    print(args)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alpha, beta, gamma = 0, 1, .1
    feature_layer = 2

    # save_dir = 'checkpoints/test.pth'
    save_dir = Path('checkpoints/dim-G')

    if(args.model == 'dim'):
      encoder, mi_estimator = get_models(alpha, beta, gamma, feature_layer)
      start_epoch = load_checkpoint({'encoder':encoder}, save_dir)
      # encoder.load_state_dict(torch.load('checkpoints/dim-L/ckpt_epoch-47_'))
    if(args.model == 'vae'):
      save_dir = Path('vae_50.pth')
      vae = VAE()
      vae.load_state_dict(torch.load(save_dir))
      encoder = vae.Encoder


    loaders = get_loaders('cifar10', batch_size=args.batch_size)
    # encoder.load_state_dict(torch.load(save_dir))
    # torch.save(encoder.state_dict(), 'checkpoints/encoder.pth')
    encoder.to(device)

    eval_model = Classifier()
    eval_model.to(device)

    optimizer = optim.Adam(eval_model.parameters(), args.lr)
    # optimizer = optim.SGD(eval_model.parameters(), args.lr, momentum=.9)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [7,50], gamma=0.5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    eval_encoder(
            encoder,
            loaders,
            eval_model,
            optimizer,
            scheduler=scheduler,
            train_epochs=args.epochs,
            batch_size=args.batch_size,
            model=args.model,
            device=device)
