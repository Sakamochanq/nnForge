from xml.parsers.expat import model
from assets.dataset import DataManager
from assets.model import Model
from assets.predict import Predict


if __name__ == "__main__":
    
    while True:
        data = DataManager()
        train_loader, val_loader, classes = data.load()
        
        model = Model().build()
        
        predictor = Predict(model, classes)
        
        image = input(' ❯ ')
        predictor.predict(image)