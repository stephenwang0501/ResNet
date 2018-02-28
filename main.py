from Trainer import ResNetTrainer

from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(1)

trainer = ResNetTrainer(resnet_size=20, batch_size=128, epoch=70, learn_rate=0.01, momentum=0.9)

trainer.set_data()

trainer.train()
