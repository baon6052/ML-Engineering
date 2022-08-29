# Generative Adversarial Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oNZSJDrkukyAKcYCErl0Y7TR9GE3eGs4?usp=sharing)

### Generator Model

```python
class Generator(nn.Module):
    def __init__(self, latent_size=100):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, NUM_CHANNELS*IMG_WIDTH*IMG_HEIGHT),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), NUM_CHANNELS, IMG_WIDTH, IMG_HEIGHT)
        return x
```

### Discriminator Model

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(NUM_CHANNELS*IMG_WIDTH*IMG_HEIGHT, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1), # 1 output for real/fake
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x
```

### Training loop

```python
def train():
    
  for epoch in range(100):
      # Iterate over some of the train dateset
      for x, t in train_loader:
          x,t = x.to(DEVICE), t.to(DEVICE)

          # Train Discriminator 
          g = G(torch.randn(x.size(0), 100).to(DEVICE))
          l_r = bce_loss(D(x).mean(), torch.ones(1)[0].to(DEVICE)) # real -> 1
          l_f = bce_loss(D(g.detach()).mean(), torch.zeros(1)[0].to(DEVICE)) #  fake -> 0
          loss_d = (l_r + l_f)/2.0
          optimiser_D.zero_grad()
          loss_d.backward()
          optimiser_D.step()
          
          # Train Generator
          g = G(torch.randn(x.size(0), 100).to(DEVICE))
          loss_g = bce_loss(D(g).mean(), torch.ones(1)[0].to(DEVICE)) # fake -> 0, but we're trying to fool D
          optimiser_G.zero_grad()
          loss_g.backward()
          optimiser_G.step()
```

### Results

Results at Epoch 100

![Results at Epoch 100](gan_results_epoch_100.png)