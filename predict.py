import os
import json
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import SimpleITK
from model import swin_tiny_patch4_window7_224 as create_model
from skimage import io, exposure, img_as_uint, img_as_float
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
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "/mnt/f/0.原始数据/0.code/swintransformerCOR/weightsCOR9/model-9,valauc0.893,valacc0.782,testauc0.785,testacc0.799.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))


    model.eval()

    # load image
    image_path = r"/mnt/f/0.原始数据/0.originNiidata/valdata/COR_MPR"
    savepath = r"/mnt/f/1.clinical"
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)
    list_data = pd.DataFrame(columns=["ID", "Proba"])
    for i in os.listdir(image_path):
        for j in os.listdir(os.path.join(image_path,i)):
            for k in os.listdir(os.path.join(image_path, i,j)):
                #img = SimpleITK.ReadImage(r"E:\data\AX\4\9000622left25.nii.gz")
                img = SimpleITK.ReadImage(os.path.join(image_path,i,j,k))
                img = SimpleITK.GetArrayFromImage(img)
                img = img.transpose(2, 1, 0)
                #img = io.imread(os.path.join(image_path, i,j,k))
                img = img.astype(np.float32)
                img = exposure.rescale_intensity(img, out_range="float32")
                #plt.imshow(img)

                # [N, C, H, W]
                img = data_transform(img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)

                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(model(img.to(device))).cpu()

                    #toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
                    #pic = toPIL(output)
                    #pic.save(r'C:\Users\Hongj\Downloads\random.jpg')



                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).detach().numpy()

                    list_newdata = pd.DataFrame({"ID": [k[:-7]], "0": [predict[0].detach().numpy()], "1": [predict[1].detach().numpy()]})
                    list_data = list_data._append(list_newdata)

                list_data.to_csv(os.path.join(savepath, "预测概率表COR.csv"),encoding="gbk", index=False)

            #print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                        # predict[predict_cla].numpy())
            #plt.title(print_res)
            #for i in range(len(predict)):
                #print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                         # predict[i].numpy()))
            #plt.show()

if __name__ == '__main__':
    main()













