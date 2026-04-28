import torch
from PIL import Image
from torchvision import transforms
from assets.config import config


class Predict:
    def __init__(self, model, classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(config.model, map_location=self.device))

        self.model.eval()

        self.classes = classes

        self.transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, pred = torch.max(output, 1)

        result = self.classes[pred.item()]
        print("\nPrediction:", result)
