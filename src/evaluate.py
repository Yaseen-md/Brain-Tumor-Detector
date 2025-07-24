import torch
from sklearn.metrics import classification_report
from data_loader import get_data_loaders
from model import get_resnet18
from utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model_path, data_dir):
    model = get_resnet18()
    model = load_model(model, model_path)
    model.to(device)
    model.eval()

    dataloaders = get_data_loaders(data_dir)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = ['glioma', 'meningioma', 'pituitary', 'non-tumor']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)