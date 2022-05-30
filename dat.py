import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os
from PIL import Image
import torch
import model as md
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
    def __getitem__(self, index):
        # Open image file, convert to numpy array and scale to [0, 1]
        #print(f"Image(self.image_files):{self.image_files}")
        image = Image.open(self.image_files[index])

        imagenp = np.array(image, dtype=np.float32) / 255
        image_name = image.filename.rsplit("\\")
        return imagenp, index, image_name[2]
        #print(np.array(image_blue.shape))
        #print(image_blue.shape[0])
        #print(image_blue.shape[1])

        # Perform normalization for each channel
        #return imagenp, index, image_name[2]
    def __len__(self):
        return len(self.image_files)

batch_size = 384 #dividable by 3 (3-image channels )
image_dataset = ImageDataset("images\images_train")
image_loader = DataLoader(image_dataset, shuffle=False, batch_size=batch_size)


y_train = pd.read_csv("data/y_train.csv",)
print(y_train)

cache = []
#cache = th.tensor(shape = (10,64,64,3))
#tensor.permute(0, 3, 1, 2)
# Iterate through the data loader

#model = torch.nn.Linear(in_features=3, out_features=3)
model = md.MModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_function = torch.nn.CrossEntropyLoss()
max_updates = 20


#'PC-3' 0 , 'U-251 MG' 1, 'HeLa' 2, 'A549' 3, 'U-2 OS' 4, 'MCF7' 5, 'HEK 293' 6, 'CACO-2' 7 and 'RT4' 8 .
#Classification Task with 9 classes - label the input correctly
y_train = np.array(y_train)
y_class = np.zeros(shape = len(y_train))

for i, y in enumerate(y_train):
    #print(y[1])
    if y[1].count('PC-3'):
        y_class[i] = 0
    if y[1].count('U-251 MG'):
        y_class[i] = 1
    if y[1].count('HeLa'):
        y_class[i] = 2 
    if y[1].count('A549'):
        y_class[i] = 3 
    if y[1].count('U-2 OS'):
        y_class[i] = 4 
    if y[1].count('MCF7'):
        y_class[i] = 5 
    if y[1].count('HEK 293'):
        y_class[i] = 6 
    if y[1].count('CACO-2'):
        y_class[i] = 7 
    if y[1].count('RT4'):
        y_class[i] = 8 
index = 0
losses = []
print(f"y_class{y_class[index]}")

epochs = 1
for epoch in range(epochs):

    with tqdm(image_loader, unit="batch") as tepoch:
        for i, (images, ids, filename) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")

            cache = []

            img = images.reshape(int(batch_size/3),3,64,64)

            targets = y_class[int(batch_size/3)*i:int(batch_size/3)*(i+1)]
            output = model(img)

            target_tensor = torch.LongTensor(targets)
            loss = loss_function(output, target_tensor)
            loss.backward()  # compute gradients (backward pass)
            optimizer.step()  # perform gradient descent update step
            optimizer.zero_grad()  # reset gradients
            losses.append(loss.detach().item())
            index += index 

            print(f"[{epoch + 1:{len(str(epochs))}d}/{epochs}] loss={loss.item():7.4f}")



