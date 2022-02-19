import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_loaders(batch_size: int, image_size: int, channels_img: int):
    transforms_ = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    train_dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms_, download=True
    )

    test_dataset = datasets.MNIST(
        root="dataset/", train=False, transform=transforms_, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return (train_loader, test_loader)
