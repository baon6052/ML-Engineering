# Variational Autoencoder - Generative Adversarial Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SoYymIggsDxhmS-6TzbfwuDR9bmqHfcM?usp=sharing)

### Encoder Model

```python
class Encoder(nn.Module):
  """
    Encoder class
    Model:
      conv1: (in_channels, 32, 32) -> (64, 32, 32)
      conv2: (64, 32, 32) -> (128, 16, 16)
      conv3: (128, 16, 16) -> (256, 8, 8)
      fc1: (256 * 8 * 8) -> (2048)
      mu: (2048) -> (nz)
      logvar: (2048) -> (nz)
  """
  def __init__(self, in_channels = 3, init_features=64, nz=100):
    super(Encoder, self).__init__()

    features = init_features
    
    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, features, kernel_size=5, stride=1, padding=2, bias=False),
      nn.BatchNorm2d(features),
      nn.ReLU(inplace=True)
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(features, features * 2, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(features * 2),
      nn.ReLU(inplace=True)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(features * 2, features * 4, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(features * 4),
      nn.ReLU(inplace=True)
    )

    self.fc1 = nn.Sequential(
      nn.Linear(256 * 8 * 8, 2048, bias=False),
      nn.BatchNorm1d(2048),
      nn.ReLU(inplace=True)
    )

    self.mu = nn.Linear(2048, nz, bias = False)
    self.logvar = nn.Linear(2048, nz, bias = False)

  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu

  def encode(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(x.size(0), -1)
    h1 = self.fc1(x)
    z_mu = self.mu(h1)
    z_var = self.logvar(h1)
    return z_mu, z_var

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return z, mu, logvar
```

### Generator Model

```python
class Generator(nn.Module):
  """
    Generator class
    Model:
      fc1: (nz) -> (256 * 8 * 8)
      upConv1: (256, 8, 8) -> (256, 16, 16)
      upConv2: (256, 16, 16) -> (128, 32, 32)
      upConv3: (128, 32, 32) -> (64, 32, 32)
      conv1: (64, 32, 32) -> (out_channels, 32, 32)
  """
  def __init__(self, out_channels = 3, nz=100):
    super(Generator, self).__init__()

    self.fc1 = nn.Sequential(
      nn.Linear(nz, 256 * 8 * 8, bias=False),
      nn.BatchNorm1d(256 * 8 * 8),
      nn.ReLU(inplace=True)
    )

    self.upConv1 = nn.Sequential(
      nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding = 1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    self.upConv2 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding = 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.upConv3 = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    self.conv1 = nn.Sequential(
      nn.Conv2d(64, out_channels, kernel_size=5, stride=1, padding=2),
      nn.Tanh()
    )

  def forward(self, z):
    out = self.fc1(z)
    out = out.view(out.size(0), 256, 8, 8)
    out = self.upConv1(out)
    out = self.upConv2(out)
    out = self.upConv3(out)
    out = self.conv1(out)
    return out
```

### Discriminator Model

```python
class Discriminator(nn.Module):
  """
    Discriminator class
    Model:
      conv1: (nc, 32, 32) -> (32, 32, 32)
      conv2: (32, 32, 32) -> (128, 16, 16)
      conv3: (128, 16, 16) -> (256, 8, 8)
      conv4: (256, 8, 8) -> (256, 8, 8)
      fc1: (256*8*8) -> (512)
      fc2: (512) -> (1)
  """

  def __init__(self, f=32):

      super(Discriminator, self).__init__()
      
      self.conv1 = nn.Sequential(
          nn.Conv2d(3, 32, 5, 1, 2, bias=False),
          nn.LeakyReLU(0.2, inplace=True),  
      )

      self.conv2 = nn.Sequential(
          nn.Conv2d(32, 128, 5, 2, 2, bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2, inplace=True),  
      )
              
      self.conv3 = nn.Sequential(
          nn.Conv2d(128, 256, 5, 2, 2, bias=False),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(0.2, inplace=True),  
      )
                      
      self.conv4 = nn.Sequential(
          nn.Conv2d(256, 256, 5, 1, 2, bias=False),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(0.2, inplace=True),  
      )

      self.fc1 = nn.Sequential(
        nn.Linear(256 * 8 * 8, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True)
      )

      self.fc2 = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
      )

  def feature(self, x):
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    return f.view(-1, 256 * 8 * 8)

  def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = x.view(-1, 256 * 8 * 8)
      x = self.fc1(x)
      x = self.fc2(x)        
      return x
```

### Training loop

```python
def train():
    y_real = torch.ones(batch_size, 1).to(DEVICE)
    y_fake = torch.zeros(batch_size, 1).to(DEVICE)

    for epoch in range(50):
        for x_real, t in train_loader:
            
            x_real, t = x_real.to(DEVICE, dtype=torch.float), t.to(DEVICE)
      
            if x_real.shape[0] == batch_size:

                # ===================
                # Train Discriminator
                # ===================
                # Get latent space vector z from real images
                z, mu, logvar = E(x_real)
                
                # Get fake images from latent space vector
                x_fake = G(z)

                # Get latent space vector from noise
                z_p = torch.randn(batch_size, 100)
                z_p = z_p.to(DEVICE)

                # Get fake images from noise
                x_p = G(z_p)

                # Compute Discriminator Loss
                y_real_loss = bce_loss(D(x_real), y_real)
                y_fake_loss = bce_loss(D(x_fake), y_fake)
                y_p_loss = bce_loss(D(x_p), y_real)
                loss_D = (y_real_loss + y_fake_loss + y_p_loss) / 3.0
                loss_D.backward(retain_graph = True)
                optimiser_D.step()


                # =============
                # Train Encoder
                # =============
                x_real, _ = next(train_iterator)
                x_real = x_real.to(DEVICE, dtype=torch.float)
                z, mu, logvar = E(x_real)
                x_fake = G(z)

                # Compute Encoder Loss
                Loss_prior = 1e-2 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                Loss_recon = 5 * l1_loss(x_fake, x_real)
                Loss_llike = l1_loss(D.feature(x_fake), D.feature(x_real))

                loss_E = Loss_prior + Loss_recon + Loss_llike
                optimiser_E.zero_grad()
                loss_E.backward()
                optimiser_E.step()


                # =========================
                # Train Generator [Decoder]
                # =========================
                x_real, _ = next(train_iterator)
                x_real = x_real.to(DEVICE, dtype=torch.float)
                z_p = torch.randn(batch_size, 100)
                z_p = z_p.to(DEVICE)
                z, mu, logvar = E(x_real)
                x_fake = G(z)
                x_p = G(z_p)

                # Compute Generator [Decoder] Loss
                y_real_loss = bce_loss(D(x_real), y_fake)
                y_fake_loss = bce_loss(D(x_fake), y_real)
                y_p_loss = bce_loss(D(x_p), y_real)
                Loss_gan_fake = (y_real_loss + y_fake_loss + y_p_loss) / 3.0

                Loss_recon = 5 * l1_loss(x_fake, x_real)
                Loss_llike = l1_loss(D.feature(x_fake), D.feature(x_real))

                loss_G = Loss_recon + Loss_llike + Loss_gan_fake
                optimiser_G.zero_grad()
                loss_G.backward()
                optimiser_G.step()
```