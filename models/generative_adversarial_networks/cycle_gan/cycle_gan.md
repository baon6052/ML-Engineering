# Cycle - Generative Adversarial Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oMcd8A83lLl1lUXb-XAQC3pBDwPWd2Y_?usp=sharing)

### Generator Model

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
              ConvBlock(channels, channels, kernel_size=3, padding=1),
              ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))
```

### Discriminator Model

```python
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
      super().__init__()
      self.conv = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
          nn.InstanceNorm2d(out_channels),
          nn.LeakyReLU(0.2),
      )

    def forward(self, x):
        return self.conv(x)
      
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size = 4,
                stride = 2,
                padding = 1,
                padding_mode = "reflect"
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))
```

### Training loop

```python
def train(total_epoch):
  loop = tqdm(train_loader, leave=True)
  
  for epoch in range(total_epoch):
      for idx, (img_A, img_B) in enumerate(loop):
          img_A, img_B = img_A.to(DEVICE), img_B.to(DEVICE)

          with torch.cuda.amp.autocast():
              fake_img_A = G_A(img_B)
              D_A_real = D_A(img_A)
              D_A_fake = D_A(fake_img_A.detach())
              D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
              D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
              D_A_loss = D_A_real_loss + D_A_fake_loss

              fake_img_B = G_B(img_A)
              D_B_real = D_B(img_B)
              D_B_fake = D_B(fake_img_B.detach())
              D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
              D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
              D_B_loss = D_B_real_loss + D_B_fake_loss

              loss_d = (D_A_loss + D_B_loss)/2
          
          optimizer_D.zero_grad()
          scaler_D.scale(loss_d).backward()
          scaler_D.step(optimizer_D)
          scaler_D.update()

          # Train Generators A and B
          with torch.cuda.amp.autocast():
              D_A_fake = D_A(fake_img_A)
              D_B_fake = D_B(fake_img_B)
              loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
              loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

              # cycle loss
              cycle_A = G_A(fake_img_B)
              cycle_B = G_B(fake_img_A)
              cycle_A_loss = l1(img_A, cycle_A)
              cycle_B_loss = l1(img_B, cycle_B)

              # identity loss
              identity_A = G_A(img_A)
              identity_B = G_B(img_B)
              identity_A_loss = l1(img_A, identity_A)
              identity_B_loss = l1(img_B, identity_B)

              loss_g = (
                  loss_G_A
                  + loss_G_B
                  + cycle_A_loss * LAMBDA_CYCLE
                  + cycle_B_loss * LAMBDA_CYCLE
                  + identity_A_loss * LAMBDA_IDENTITY
                  + identity_B_loss * LAMBDA_IDENTITY
              )


          optimizer_G.zero_grad()
          scaler_G.scale(loss_g).backward()
          scaler_G.step(optimizer_G)
          scaler_G.update()
```