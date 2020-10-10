import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# functions to show an image
def imshow(img, fn):
    plt.clf()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(fn)


def download_data(data_path):
    CIFAR10(root=data_path, train=True, download=True)
    os.remove(os.path.join(data_path, CIFAR10.filename))


def train(
    data_path,
    state_path,
    model_path,
    num_workers=2,
    train_batch_size=40,
    test_batch_size=4,
    epochs=5,
):
    # ##################### 1. Loading and normalizing CIFAR10

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(root=data_path, train=True, download=False, transform=transform)
    # trainset = torch.utils.data.ConcatDataset([trainset]*5)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if False:
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images), os.path.join(state_path, "11.png"))
        # print labels
        print(" ".join("%5s" % classes[labels[j]] for j in range(4)))

    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    net.to(device)

    # ##################### 3. Define a Loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=0.001 * train_batch_size / 40, momentum=0.9
    )

    # ##################### 4. Train the network

    print_frq = 2000 * 40 // train_batch_size
    num_batches = len(trainloader)
    print(f"Num batches:{num_batches}")
    torch.save(net.state_dict(), os.path.join(state_path, "cifar_net.pth"))

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch}...")
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (
                i % print_frq == (print_frq - 1) or i == num_batches - 1
            ):  # print every 10 mini-batches
                torch.save(net.state_dict(), os.path.join(state_path, "cifar_net.pth"))
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        test(
            data_path, state_path, num_workers, test_batch_size, net=net, device=device
        )

    print("Finished Training")
    # save the model
    torch.save(net.state_dict(), os.path.join(model_path, "cifar_net.pth"))


# ##################### 2. Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ##################### 5. Test the network on the test data
def test(
    data_path,
    state_path,
    num_workers=2,
    test_batch_size=4,
    net=None,
    test_classes=False,
    device=None,
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    testset = CIFAR10(root=data_path, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not Net:
        print(f"Testing on {device}")
        net = Net()
        net.to(device)
        PATH = os.path.join(state_path, "cifar_net.pth")
        net.load_state_dict(torch.load(PATH))

    if test_classes:
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        # print images
        imshow(torchvision.utils.make_grid(images), os.path.join(state_path, "22.png"))
        print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)

        print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

    if test_classes:
        class_correct = list(0.0 for i in range(10))
        class_total = list(0.0 for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print(
                "Accuracy of %5s : %2d %%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )


def parseArgs():
    parser = argparse.ArgumentParser(description="Cifar 10 training")

    # Worker type
    parser.add_argument("--download_only", action="store_true", default=False)

    # Distribution
    parser.add_argument("--backend", default="")
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--host_rank", type=int, default=0)

    # Paths
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--state_path", default="./model")
    parser.add_argument("--model_path", default="./model")

    # Hyperparameters
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=40)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # 1. Get the worker configuration and
    worker_config = None
    if "SAGEMAKER_JOB_NAME" in os.environ:
        from worker_toolkit import worker_lib

        worker_config = worker_lib.WorkerConfig(per_instance_state=False)

    args = parseArgs()

    if args.download_only:
        # 2a. Update the parsed command line arguments from the configuration
        if worker_config:
            args.data_path = worker_config.state
        download_data(args.data_path)
        return

    # 2b. Update the parsed command line arguments from the configuration
    if worker_config:
        worker_config.initMultiWorkersState()
        # Update paths
        args.state_path = worker_config.instance_state
        args.data_path = worker_config.channel_cifar_data
        args.model_path = worker_config.model_dir
        # Update distribution parameters
        args.num_nodes = worker_config.num_nodes
        args.host_rank = worker_config.host_rank

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.state_path, exist_ok=True)

    if args.distributed:
        logger.info("*** Distributed training")
        # Initialize the distributed environment.
        if not worker_config:
            # 3b. The are set automatically when using a PyTorch framework
            #   see https://github.com/aws/sagemaker-pytorch-training-toolkit/blob/
            #           master/src/sagemaker_pytorch_container/training.py
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "7777"

        os.environ["WORLD_SIZE"] = str(args.num_nodes)
        dist.init_process_group(backend=args.backend, rank=args.host_rank)

    else:
        logger.info("*** Single node training")

    train(
        args.data_path,
        args.state_path,
        args.model_path,
        num_workers=args.num_workers,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
