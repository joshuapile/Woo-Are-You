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
from trainer import trainingclass

class birdmodel:

  def __init__(self,contentpath):
         self=self
         self.contentpath=contentpath
         self.trainer = trainingclass()

  def savetrainedmodel(self):
    torch.cuda.empty_cache()
    datatorchs = Path(self.contentpath)
    images= datatorchs / "PyTorchBirdData"

    datatransform=transforms.Compose([transforms.Resize(size=(256,256)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor() ,
                                      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                      ])

    trainimages = images / "train"
    testimages =  images / "test"

    trainingdata = datasets.ImageFolder(root=trainimages,
                                              transform=datatransform)
    testingdata = datasets.ImageFolder(root=testimages,
                                              transform=datatransform)

    BATCH=64
    traindataloader=DataLoader(dataset=trainingdata,batch_size=BATCH,num_workers=1,shuffle=True)

    testdataloader=DataLoader(dataset=testingdata,batch_size=BATCH,num_workers=1,shuffle=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = nn.Linear(2048, 365)

    classes=trainingdata.classes
    lossfunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    self.trainer.train_model(model,{"train":traindataloader,"val":testdataloader} ,optimizer , num_epochs=30, file_name = "model" ,show_plot=False, is_inception=False, feature_extract=True)

    self.torch.save(model, self.contentpath+"birdclassiferresnet50_e3.pth")
    mod = torch.load(self.contentpath+'birdclassiferresnet50_e3.pth')
    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testdataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = mod(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


    dataiter = iter(testdataloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # print images
    self.trainer.imshow(torchvision.utils.make_grid(images).cpu())
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))

    outputs = mod(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(8)))

