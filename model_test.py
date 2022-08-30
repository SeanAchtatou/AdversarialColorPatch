import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader

from model_Road import *

def transformsXXX(path,bb,transforms):
    x = cv2.cvtColor(path.astype(np.float32), cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5:
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x




model = torch.load("Models/pyTorch_Model.h5")
for i in range(59,700):
    im = read_image(f'images/road{i}.png')
    im = cv2.resize(im, (int(1.49*300), 300))
    cv2.imshow("Im",im)
    cv2.imwrite('images_resized/resized_road789.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    f = time.time()
    for i in range(1,300):
        test_ds = RoadDataset(pd.DataFrame([{'path':'images_resized/resized_road789.jpg'}])['path'],pd.DataFrame([{'bb':np.array([0,0,0,0])}])['bb'],pd.DataFrame([{'y':[0]}])['y'])
        print(i)
    ff = time.time()
    print(ff-f)
    x, y_class, y_bb = test_ds[0]
    f = time.time()
    for i in range(0,300):
        w = np.rollaxis(normalize(transformsXXX(im,np.array([0,0,0,0]),True)),2)
    ff = time.time()
    print(ff-f)
    xx = torch.FloatTensor([x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x])

    print(len(xx))
    f = time.time()
    out_class, out_bb = model(xx)
    ff = time.time()
    print(ff-f)
    f = time.time()
    for i in range(0,len(xx),32):
        out_class, out_bb = model(xx[i:i+32])
        print(i+32)
    ff = time.time()
    print(ff-f)
    input("Wait...")
    prob = F.softmax(out_class,1)
    pred = torch.max(out_class, 1)[1]
    print(prob)

    print([float(max(prob[i])) for i in range(len(prob))[:2]])
    print(prob[np.array([i for i in range(len(prob))][:5]),prob[:5]])


    if pred == 0:
        print("Speed")
    if pred == 1:
        print("Stop")
    if pred == 2:
        print("Cross")
    if pred == 3:
        print("Traffic")


    bb_hat = out_bb.detach().cpu().numpy()
    bb_hat = bb_hat.astype(int)
    show_corner_bb(im, bb_hat[0])
