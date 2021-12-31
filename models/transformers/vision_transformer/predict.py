import urllib

import numpy as np
import torch
from PIL import Image

k = 10

imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("model.pth")
model.eval()

# get in range -1, 1
img = (np.array(Image.open("cat.png")) / 128) - 1
input = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(input)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")
