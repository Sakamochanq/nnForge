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

        # 勾配を計算しない
        with torch.no_grad():
            # 予測
            output = self.model(image)
            # 確率の計算
            probabilities = torch.softmax(output, dim=1)
            _, pred = torch.max(probabilities, 1)

        result = self.classes[pred.item()]
        
        
        print(f"\nPredicted: {result}\n")
        for idx, class_name in enumerate(self.classes):
            percentage = probabilities[0][idx].item() * 100
            print(f"{class_name}: {percentage:.2f}%")
        print("")
