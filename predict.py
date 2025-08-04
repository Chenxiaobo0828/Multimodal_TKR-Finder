import os
import json
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import SimpleITK
from swim_model import swin_tiny_patch4_window7_224 as create_model
from skimage import io, exposure
import numpy as np
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(img_size),
                                   transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                   ])

    # read class_indict
    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = r"C:\Users\Englishday\Desktop\0706knee\github_knee\weights\wCOR.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)

    model.eval()

    # load image
    image_path = r"C:\Users\Englishday\Desktop\0706knee\International test data from OAI"
    savepath = r"C:\Users\Englishday\Desktop\0706knee\save"
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)
    list_data = pd.DataFrame(columns=["ID", "Proba"])
    for i in os.listdir(image_path):
        for j in os.listdir(os.path.join(image_path,i)):
            for k in os.listdir(os.path.join(image_path, i,j)):
                img = SimpleITK.ReadImage(os.path.join(image_path,i,j,k))
                img = SimpleITK.GetArrayFromImage(img)
                img = img.transpose(2, 1, 0)
                img = img.astype(np.float32)
                img = exposure.rescale_intensity(img, out_range="float32")
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)

                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)

                    list_newdata = pd.DataFrame({"ID": [k[:-7]], "0": [predict[0].detach().numpy()], "1": [predict[1].detach().numpy()]})
                    list_data = list_data._append(list_newdata)

                list_data.to_csv(os.path.join(savepath, "COR.csv"),encoding="gbk", index=False)


if __name__ == '__main__':
    main()













