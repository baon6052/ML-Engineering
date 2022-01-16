import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64
CHANNELS_IMG = 1
IMAGE_SIZE = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
        ),
    ]
)

train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms, download=True
)

test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms, download=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
