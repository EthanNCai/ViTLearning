import torch

model_path = "../saved_model/imgnet_pretrain_epoch_11_acc_36_.pth"
model = torch.load(model_path)

for k, v in model.items():
    print(k)
    print(v)
    print("=" * 50)
