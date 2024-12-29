# -*- coding: utf-8 -*-
"""RESNET18_birds.ipynb

Original file using the Google Colab
"""

!pip install jovian --upgrade --quiet

# Commented out IPython magic to ensure Python compatibility.
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline
from tqdm.notebook import tqdm
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split

project_name='pytorch-zero-to-gan-course-project'

"""Data Prepation"""

from google.colab import drive
drive.mount('/content/drive')

import os

# Adjust these paths to match your Google Drive structure
DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks"
TRAIN_DIR = "/content/drive/MyDrive/Colab Notebooks/train"
VAL_DIR = "/content/drive/MyDrive/Colab Notebooks/valid"

# Verify the directories exist
print(f"Train directory exists: {TRAIN_DIR}")
print(f"Validation directory exists: {VAL_DIR}")

# List contents (first 5 items) if directories exist
if os.path.exists(TRAIN_DIR):
    print(f"Contents of train directory: {os.listdir(TRAIN_DIR)[:5]}")
if os.path.exists(VAL_DIR):
    print(f"Contents of validation directory: {os.listdir(VAL_DIR)[:5]}")

transform_ds = T.Compose([
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

train_ds = torchvision.datasets.ImageFolder(root=TRAIN_DIR,
                                           transform=transform_ds)

num_classes= len(train_ds.classes)
num_classes

val_ds = torchvision.datasets.ImageFolder(root=VAL_DIR,
                                         transform=transform_ds)

batch_size= 512

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

def show_images(train_dl):
    for images, labels in train_dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:32], nrow=8).permute(1,2,0))
        break

"""Move to GPU"""

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)

    def __len__(self):
        return len(self.dl)

device = get_device()
device

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

"""Defining Model"""

def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, epochs, result):
        print("Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch+1, epochs, result["lrs"][-1], result["train_loss"], result["val_loss"], result["val_acc"]))

class model(ImageClassificationBase):
     def __init__(self, num_classes):
         super().__init__()
         self.network = models.resnet18(pretrained=True)
         number_of_features = self.network.fc.in_features
         self.network.fc = nn.Linear(number_of_features, num_classes)

     def forward(self, xb):
         return self.network(xb)

     def freeze(self):
         for param in self.network.parameters():
             param.requires_grad= False
         for param in self.network.fc.parameters():
             param.requires_grad= True

     def unfreeze(self):
         for param in self.network.parameters():
            param.requires_grad= True

model = to_device(model(num_classes), device)

"""Training Phase"""

@torch.no_grad()
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, weight_decay=0, grad_clip=None,
                 opt_func=torch.optim.Adam):

    torch.cuda.empty_cache()

    history = []
    opt = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs=epochs,
                                                   steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        model.train()
        train_loss = []
        lrs = []
        for batch in tqdm(train_dl):
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            opt.step()
            opt.zero_grad()

            lrs.append(get_lr(opt))
            sched.step()

        result = evaluate(model, val_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, epochs, result)
        history.append(result)
    return history

history = [evaluate(model, val_dl)]
history

"""Training On Current Dataset"""

epochs = 10
max_lr = 10e-5
grad_clip = 0.1
weight_decay = 10e-4
opt_func=torch.optim.Adam

model.freeze()

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl, weight_decay=weight_decay,
                        grad_clip=grad_clip, opt_func=opt_func)

model.unfreeze()

epochs = 10
max_lr = 10e-5
grad_clip = 0.1
weight_decay = 10e-4
opt_func = torch.optim.Adam

history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                        weight_decay=weight_decay, grad_clip=grad_clip,
                        opt_func=opt_func)

"""Test"""

TEST_DIR = "/content/drive/MyDrive/Colab Notebooks/test"

transform_test = T.Compose([
    T.Resize((128,128)),
    T.ToTensor()
])
test_ds = torchvision.datasets.ImageFolder(root=TEST_DIR,
                                          transform=transform_test)
def prediction(model, images):
    xb = to_device(images.unsqueeze(0), device)
    out = model(xb)
    _, preds = torch.max(out, dim=1)
    predictions = test_ds.classes[preds[0].item()]
    return predictions
images, labels = test_ds[0]
print("Label: ", test_ds.classes[labels])
print("Prediction: ", prediction(model, images))
plt.imshow(images.permute(1,2,0))

"""Performance"""

accuracy = [x["val_acc"] for x in history]
plt.plot(accuracy, "-rx")
plt.title("Accuracy vs number of epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

val_loss = [x["val_loss"] for x in history]
train_loss = [x.get("train_loss") for x in history]
plt.plot(val_loss, "-bx")
plt.plot(train_loss, "-gx")
plt.title("Losses vs number of epochs")
plt.legend(["Validation loss", "Train loss"])
plt.xlabel("Epochs")

torch.save(model.state_dict(), 'BirdsResNet18.pth')

jovian.commit(project=project_name, output=['BirdsResNet18.pth'])

jovian.log_metrics(train_loss=history[-1]['train_loss'],
                   val_loss=history[-1]['val_loss'],
                   val_acc=history[-1]['val_acc'])
