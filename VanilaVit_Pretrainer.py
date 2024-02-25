import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import StepLR
from vit_pytorch import ViT
import time
import random
import os
import numpy as np
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_train = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225))
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225))
])

batch_size = 40
epochs = 60
lr = 3e-5
gamma = 0.7
step = 1
seed = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform_val)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

model = ViT(
    image_size=256,
    patch_size=16,
    num_classes=10,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

model_file = None
epochs_pretrained = 0

"""
load from pretrain
"""
pre_weights = torch.load('./saved_model/imgnet_pretrain_epoch_11_acc_36_.pth')
pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
"""
freeze pretrained weight
"""
# for param in model.features.parameters():
#    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step, gamma=gamma)


def topk_accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size_ = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    acc_list = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc_list.append(correct_k.mul_(1.0 / batch_size_))
    return acc_list


def main():
    for epoch in range(epochs_pretrained, epochs):
        start_time = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        top1_accuracy = 0
        top5_accuracy = 0

        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = topk_accuracy(output, label, topk=(1, 5))
            epoch_accuracy += acc[0] / len(train_loader)
            top1_accuracy += acc[0]
            top5_accuracy += acc[1]
            epoch_loss += loss / len(train_loader)

            print(
                f"\rEpoch:{epoch}/{epochs} Step : {i}/{len(train_loader)} - step_loss : {loss.item():.4f}", end=''
            )

        scheduler.step()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            top1_val_accuracy = 0
            top5_val_accuracy = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = topk_accuracy(val_output, label, topk=(1, 5))
                epoch_val_accuracy += acc[0] / len(test_loader)
                top1_val_accuracy += acc[0]
                top5_val_accuracy += acc[1]
                epoch_val_loss += val_loss / len(test_loader)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"\nEpoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : "
            f"{epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - Time: {epoch_time:.2f} seconds"
        )
        print(
            f"Top-1 Accuracy: {top1_val_accuracy / len(test_loader):.4f} - Top-5 Accuracy: {top5_val_accuracy / len(test_loader):.4f}"
        )

        # 保存模型awd
        torch.save(model.state_dict(), f"./saved_model/cifar10_fine_tune_epoch_{epoch + 1}_acc_{int(epoch_val_accuracy * 100)}_.pth")


if __name__ == '__main__':
    main()
