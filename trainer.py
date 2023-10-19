import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import requests
import zipfile
import copy
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms2
import numpy as np
import torch.optim as optim

class trainingclass:

  def __init__(self):
         self=self

  def imshow(self,img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

  def train_model(self,model, dataloaders, optimizer_ft, num_epochs=25, file_name = "model" ,show_plot=False, is_inception=False, feature_extract=True):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_ft = model.to(device)
            if feature_extract:
                params_to_update = []
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t",name)
            else:
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        print("\t",name)

            # Observe that all parameters are being optimized
            criterion = nn.CrossEntropyLoss()

            since = time.time()

            val_acc_history = []

            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            train_losses = []
            val_losses = []
            train_accuracies = []
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                      

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer_ft.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            if is_inception and phase == 'train':
                                
                                outputs, aux_outputs = model(inputs)
                                loss1 = criterion(outputs, labels)
                                loss2 = criterion(aux_outputs, labels)
                                loss = loss1 + 0.4*loss2
                            else:
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer_ft.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        val_acc_history.append(epoch_acc)
                        val_losses.append(epoch_loss)
                    else:
                        train_accuracies.append(epoch_acc)
                        train_losses.append(epoch_loss)


            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights
            model.load_state_dict(best_model_wts)
            return model, val_acc_history

