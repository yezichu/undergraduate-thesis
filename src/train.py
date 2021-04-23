from Trainer import Trainer
from Dataset.BratsDataset import BratsDataset
from model.H_Resunet import H_ResUnet
from loss.loss_metric import BCEDiceLoss
from torchvision import transforms
from Dataset.color_augmentations import RandomIntensityScale, RandomIntensityShift, RandomGaussianNoise
from Dataset.spatial_augmentations import RandomMirrorFlip, RandomRotation90

transform = transforms.Compose([
    RandomIntensityScale(p=0.2),
    RandomMirrorFlip(p=0.3),
    RandomRotation90(p=0.3),
])

data_path = '/home/mimt/jupyter/wy/train'
nodel = H_ResUnet(80).to('cuda')

trainer = Trainer(net=nodel,
                  dataset=BratsDataset,
                  criterion=BCEDiceLoss(),
                  lr=0.001,
                  accumulation_steps=4,
                  batch_size=1,
                  transform=transform,
                  num_epochs=100,
                  warm_up_epochs=5,
                  data_path=data_path)

trainer.run()
