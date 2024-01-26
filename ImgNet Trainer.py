from MLclf import MLclf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from vit_pytorch import SimpleViT


MLclf.miniimagenet_download(Download=False)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2,
                                                                                 seed_value=None, shuffle=True,
                                                                                 transform=transform,
                                                                                 save_clf_data=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0)

model = SimpleViT(
    image_size=84,
    patch_size=4,
    num_classes=100,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048
).cuda()

epochs = 90

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        total_step = len(train_loader)
        inputs = inputs.cuda()
        labels = torch.tensor(labels).to(torch.long).cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
        running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Step {i}/{total_step} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

model.eval()
test_loss = 0.0
test_correct_predictions = 0
test_total_predictions = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        test_correct_predictions += (predicted == labels).sum().item()
        test_total_predictions += labels.size(0)

        test_loss += loss.item()

test_epoch_loss = test_loss / len(test_loader)
test_epoch_accuracy = test_correct_predictions / test_total_predictions

print(f"Test Loss: {test_epoch_loss:.4f} - Test Accuracy: {test_epoch_accuracy:.4f}")
