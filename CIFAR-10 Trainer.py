import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.optim.lr_scheduler import StepLR
from vit_pytorch import SimpleViT
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(32, antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64
epochs = 35
lr = 3e-5
gamma = 0.7

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform_val)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

model = SimpleViT(
    image_size=32,
    patch_size=2,
    num_classes=10,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


def main():
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        epoch_accuracy = 0

        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            print(
                f"Epoch:{epoch}/{epochs} Step : {i}/{len(train_loader)} - step_loss : {loss.item():.4f}"
            )

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(test_loader)
                epoch_val_loss += val_loss / len(test_loader)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : "
            f"{epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} - Time: {epoch_time:.2f} seconds\n"
        )

        # 保存模型
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


if __name__ == '__main__':
    main()