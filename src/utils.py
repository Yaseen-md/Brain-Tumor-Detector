import torch
import torch.nn.functional as F
from torchvision import transforms

def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def load_model(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        _, predicted_idx = torch.max(probs, 1)
    return probs[0].cpu().numpy(), predicted_idx.item()