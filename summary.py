from torchinfo import summary
from models.VNet import VNet
from configs.config import Config
model = VNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
# summary(model, input_size=(1, 1, 64, 64, 64))
print(model)