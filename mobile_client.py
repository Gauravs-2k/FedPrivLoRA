import flwr as fl
import torch
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as data

class MobileClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = mobilenet_v2(num_classes=10)
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.train_loader = data.DataLoader(self.train_data, batch_size=32, shuffle=True)
        self.val_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.val_loader = data.DataLoader(self.val_data, batch_size=32, shuffle=False)
        
    def get_parameters(self, config):
        return [val.cpu().detach().numpy() for val in self.model.state_dict().values()]
    
    def fit(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1): 
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config), len(self.train_data), {}
    
    def evaluate(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return 1 - accuracy, len(self.val_data), {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_client(server_address="localhost:8081", client=MobileClient().to_client())