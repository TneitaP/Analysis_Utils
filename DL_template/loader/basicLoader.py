import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt

from enum import IntEnum 


class ToyDatasetName(IntEnum):
    MNIST = 0,
    FashionMNIST = 1, 
    #CIFAR10 = 2, 

Dataset_Dic = {
    ToyDatasetName.MNIST: torchvision.datasets.MNIST, 
    ToyDatasetName.FashionMNIST: torchvision.datasets.FashionMNIST, 
}


def toy_dataset(pDataset_name:ToyDatasetName, pRoot, pIstrain, pTransform, pDownload):

    if pTransform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
            # transforms.Normalize([0.5], [0.5])
            ] # range [0, 1.0] -> [-1.0,1.0]
            )
    else:
        transform = pTransform
    return Dataset_Dic[pDataset_name](root = pRoot, train = pIstrain, transform = transform, download = pDownload)


def view_tensor(p_img_Tsor):
    p_img_Tsor = p_img_Tsor / 2 + 0.5     # unnormalize
    img_Arr = p_img_Tsor.numpy()
    plt.imshow(np.transpose(img_Arr, (1, 2, 0)))
    plt.show()



def test_dataloading():
    data_root = "datasets" if os.path.isdir(os.path.join(os.getcwd(), "datasets")) else "../datasets"
    gm_dataset = toy_dataset(ToyDatasetName.MNIST, data_root, True, None, True)

    gm_trainloader = torch.utils.data.DataLoader(
        dataset = gm_dataset, 
        batch_size= 16,
        shuffle= True,
        num_workers= 2
    )

    for img_Tsor_bacth_i, label_Tsor_bacth_i in gm_trainloader:
        print("img tensor shape: ", img_Tsor_bacth_i.shape)
        view_tensor(torchvision.utils.make_grid(
                        tensor = img_Tsor_bacth_i, 
                        nrow= 4)
            )

if __name__ == "__main__":

    test_dataloading()

