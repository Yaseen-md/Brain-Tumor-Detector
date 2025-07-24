import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir='./datasets', batch_size=32, augment=True):
    """
    Returns train, validation, and test DataLoaders.
    
    Args:
        data_dir (str): Path to dataset directory containing 'train', 'val', 'test' subfolders.
        batch_size (int): Batch size for DataLoader.
        augment (bool): Whether to apply data augmentation on training set.

    Returns:
        dict: Dictionary of DataLoaders {'train': ..., 'val': ..., 'test': ...}
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if augment:
        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    transform_val_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(os.path.join('../dataset/train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join('../dataset/val'), transform=transform_val_test)
    test_dataset = datasets.ImageFolder(os.path.join('../dataset/test'), transform=transform_val_test)

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size),
        'test': DataLoader(test_dataset, batch_size=batch_size)
    }

    return loaders