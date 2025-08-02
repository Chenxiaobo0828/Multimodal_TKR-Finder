import os
import math
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from swin_model import swin_tiny_patch4_window7_224 as create_model
from skimage import io, exposure, img_as_uint, img_as_float
import SimpleITK
import cv2
class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result

def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img

def main(path,savepath,target_category,weights_path,load):
    # 注意输入的图片必须是32的整数倍
    # 否则由于padding的原因会出现注意力飘逸的问题
    img_size = 224
    assert img_size % 32 == 0

    model = create_model(num_classes=2)
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    weights_path = weights_path
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

    target_layers = [model.norm]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(img_size),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])
    # load image
    img_path = path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    if load==0:
        img = io.imread(img_path)[:, :, :3]
    if load==1:
        img = SimpleITK.ReadImage(img_path)
        img = SimpleITK.GetArrayFromImage(img)
        img = img[int(img.shape[0] / 2 - 1):int(img.shape[0] / 2 + 2), :, :]
        img = img.transpose(2, 1, 0)

    img = img.astype(np.float32)
    img = exposure.rescale_intensity(img, out_range="float32")
    rat = img.shape[1]/256
    newhigh = int(img.shape[0]/rat)
    newimg = cv2.resize(img, dsize=(256,newhigh))
    higrat = int((newhigh-224)/2)
    newimg = newimg[higrat:higrat+224,16:240]


    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))
    target_category = target_category  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(newimg, grayscale_cam,use_rgb=True)
    if load==1:
        visualization = np.flip(visualization, axis=1)#镜面翻转
        visualization = np.rot90(visualization,1)#向左旋转

    plt.imsave(savepath,visualization)
    #plt.imshow(visualization)
    #plt.show()

path = r"/mnt/d/kneesoft/bulaiying.nii.gz" #病人的数据路径,具体为 xxxx/xx.nii.gz
motai = "COR_MPR" #病人的序列
label = 1 #想要绘制的class

savepath = r"/mnt/d/kneesoft/testCAM" #CAM的保存路径


if motai == "Xray": #判断模态
   weights_path ="/mnt/d/kneesoft/weights/wXRAY.pth"
if motai == "SAG_IW_TSE":
   weights_path = "/mnt/d/kneesoft/weights/wSAG.pth"
if motai == "COR_MPR":
   weights_path = "/mnt/d/kneesoft/weights/wCOR.pth"


if os.path.exists(os.path.join(savepath,motai)) is False:
    os.makedirs(os.path.join(savepath,motai))
name = os.path.split(path)[1]

if motai == "Xray":
    thesave = os.path.join(savepath,motai,name[:-4]+".jpg")#输入为jpg格式的Xray，非DICOM
    main(path,thesave,label,weights_path,load=0)
else:
    thesave = os.path.join(savepath,motai,name[:-7] + ".jpg") #输入形式为 .nii.gz
    main(path, thesave,label, weights_path,load=1)

