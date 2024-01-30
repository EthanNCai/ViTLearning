from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import glob

normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    normalize
])
train_dataset = ImageFolder('../imagenet1k/train', transform=transform)
test_dataset = ImageFolder('../imagenet1k/val', transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0)

for i, (data, label) in enumerate(train_loader):
        print(i)
        print(data)
        print(label)
        break
for i, (data, label) in enumerate(test_loader):
        print(i)
        print(data)
        print(label)
        break
