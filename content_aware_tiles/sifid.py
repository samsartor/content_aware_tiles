from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import torch
import numpy as np

inception = None

def sifid(img1: torch.Tensor):
    global inception
    if inception is None:
        inception = InceptionV3([2]).eval().to(img1.device)

    assert img1.shape[0] == 1
    pred1 = inception(img1.to(torch.float32))[0]
    act1 = pred1.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(pred1.shape[2]*pred1.shape[3],-1)
    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1, rowvar=False)


    def calculate(img2: torch.Tensor):
        assert inception is not None
        assert img2.shape[0] == 1
        pred2 = inception(img2.to(torch.float32))[0]
        act2 = pred2.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(pred2.shape[2]*pred2.shape[3],-1)
        mu2 = np.mean(act2, axis=0)
        sigma2 = np.cov(act2, rowvar=False)

        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return calculate
