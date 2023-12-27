from discriminator import Discriminator, load_image
import torch
from PIL import Image
from torchvision import transforms

def evaluate_image(model, image, label):
    with torch.no_grad():
        output = model(image)
        predicted_label = 'Real' if output.item() > 0.5 else 'Fake'
        print(f"Label: {label}, Predicted: {predicted_label}, Confidence: {output.item()}")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

real_artwork = load_image('/home/tjddms9376/cv/data/real/1.png').unsqueeze(0)  # Add batch dimension
generated_artwork = load_image('/home/tjddms9376/cv/data/fake/3.png').unsqueeze(0)

loaded_model = Discriminator()
loaded_model.load_state_dict(torch.load('/home/tjddms9376/cv/discriminator/pretrained/model_latest.pth'))
loaded_model.eval()  

evaluate_image(loaded_model, generated_artwork, 'Fake')
evaluate_image(loaded_model, real_artwork, 'Real')
