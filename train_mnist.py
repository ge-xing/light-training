
from torchvision import datasets, transforms
import torch.nn as nn 
import torch
import torch.nn.functional as F 
from light_training.trainer import Trainer
from monai.utils import set_determinism
set_determinism(123)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
class MNISTTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.model = Net()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        self.loss_func = nn.CrossEntropyLoss()

    def training_step(self, batch):
        image, label = batch

        pred = self.model(image)
        loss = self.loss_func(pred, label)
        self.log("train_loss", loss, self.global_step)

        return loss

    def validation_step(self, batch):

        image, label = batch
        pred = self.model(image)
        loss = self.loss_func(pred, label)
        pred = pred.argmax(dim=-1)
        right_num = (pred == label).sum()
        self.log("val_loss", loss, self.global_step)

        return [loss, right_num] 

    def validation_end(self, mean_val_outputs, val_outputs):
       
        val_loss, acc = mean_val_outputs

        self.log("val_acc", acc, self.epoch)

        if self.local_rank == 0:
            print(f"val loss is {val_loss}")
            print(f"acc is {acc}")


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset1 = datasets.MNIST('/home/xingzhaohu/sharefs/datasets/', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('/home/xingzhaohu/sharefs/datasets/', train=False,
                    transform=transform)

trainer = MNISTTrainer(env_type="DDP",
                max_epochs=10,
                batch_size=128,
                device="cuda:1",
                val_every=1,
                num_gpus=4,
                training_script=__file__)


trainer.train(train_dataset=dataset1, val_dataset=dataset2)

