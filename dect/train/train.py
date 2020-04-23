import torch
from model.model import get_model
from config.config import *
from dataset.CrowdHumanGenerator import CrowdHumanGenerator
from torch.utils.data import DataLoader
from util.Engine import train_one_epoch, evaluate
from util import Transforms as T

train_dataset_path      = "data"
validation_dataset_path = "data"
#resume_net_path         = "weight/efficient_rcnn_2.pth"
resume_net_path         = None
weight_save_folder      = "weight"
lr              = 0.005
momentum        = 0.9
weight_decay    = 0.0005
num_epochs      = 10
VERSION_FAST    = 49

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
train_dataset = CrowdHumanGenerator(path=train_dataset_path, type="train", config=None, transform=get_transform(train=True))
validation_dataset = CrowdHumanGenerator(path=validation_dataset_path, type="validation", config=None, transform=get_transform(train=False))

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=10, collate_fn=collate_fn)

test_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=10, collate_fn=collate_fn)

model = get_model(VERSION_FAST)

if resume_net_path is not None:
    print("load weights from {}".format(resume_net_path))
    model.load_state_dict(torch.load(resume_net_path))

model.eval()
print('Finished loading model!')
print(model)

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print("device = ", device)
model.to(device)

params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    print("device = ", device)
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()

    torch.save(model.state_dict(), "{}/efficient_rcnn_".format(weight_save_folder) + str(epoch) + ".pth")

    # evaluate on the test dataset
    evaluate(model, test_loader, device=device)