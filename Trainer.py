from MLclf import MLclf
import torch
import torchvision.transforms as transforms
from vit_pytorch import SimpleViT

MLclf.miniimagenet_download(Download=False)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2,
                                                                                 seed_value=None, shuffle=True,
                                                                                 transform=transform,
                                                                                 save_clf_data=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=0)

model = SimpleViT(
    image_size=84,
    patch_size=4,
    num_classes=100,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048
)

epochs = 200

for epoch in range(epochs):
    for i, (input_, labels) in enumerate(train_loader):
        result = model(input_)
        print(result.shape)
        break
    break


