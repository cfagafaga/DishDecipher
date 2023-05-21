import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.backbone = torchvision.models.resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total +=images.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = correct / total

    return train_loss, train_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += images.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total

    return test_loss, test_acc

def main():
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据变换
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])   

    # 加载数据集
    train_data = torchvision.datasets.ImageFolder('train', transform=data_transforms)       
    test_data = torchvision.datasets.ImageFolder('test', transform=data_transforms)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)       
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    # 加载ResNet34模型
    model = FoodClassifier(num_classes=10).to(device)

    # 定义优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 定义学习率调度
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练模型
    best_acc = 0.0
    for epoch in range(20):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        # 更新学习率
        lr_scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'food_classifier.pth')

if __name__ == '__main__':
    main()