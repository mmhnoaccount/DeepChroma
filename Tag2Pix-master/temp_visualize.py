import numpy as np
import cv2
import os
import re
import torch.nn as nn
import torch

def tryint(s):                       #将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):                #将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):    #以分割后的list为单位进行排序
    return sorted(v_list, key=str2int)

def build_fullSize_Pic(PicPackage):
    #height = 1352 // 2
    #width = 1352 // 2
    #height = 480 // 2
    #width = 832 // 2
    height = 1080 // 2
    width = 1920 // 2

    A = np.zeros((height, width), dtype=float)

    image_paths = os.listdir(PicPackage)
    image_paths = sort_humanly(image_paths)
    # print(image_paths)
    for j in range(width // 32):
        for i in range(height // 32):
            f = open(PicPackage + '/' + image_paths[j * (height // 32) + i])
            lines = f.readlines()
            block_row = i * 32
            for line in lines:
                per_line = line.strip('\n').split(' ')
                A[block_row, j * 32: j * 32 + 32] = per_line[:-1]
                block_row += 1

    A = A / ((1 << 10) - 1)
    A = A * 255
    A = A.astype(np.uint8)
    return A


def visualize_results(epoch, fix=True):
    #Luma_PicPackage = r'E:\visualize_set\train_visualize_22\luma_32_32\CCLM'
    #Cb_PicPackage = r'E:\visualize_set\train_visualize_22\Cb_32_32\org'
    #Cr_PicPackage = r'E:\visualize_set\train_visualize_22\Cr_32_32\org'

    Luma_PicPackage = r'D:\tashikete\debug_visualize\luma_32_32\CCLM'
    Cb_PicPackage = r'D:\tashikete\debug_visualize\Cb_32_32\pred'
    Cr_PicPackage = r'D:\tashikete\debug_visualize\Cr_32_32\pred'
    #Cb_PicPackage = r'D:\tashikete\debug_visualize\Cb_32_32\CCLM'
    #Cr_PicPackage = r'D:\tashikete\debug_visualize\Cr_32_32\CCLM'
    #Cb_PicPackage = r'D:\tashikete\debug_visualize\Cb_32_32\org'
    #Cr_PicPackage = r'D:\tashikete\debug_visualize\Cr_32_32\org'


    Luma = build_fullSize_Pic(Luma_PicPackage)

    Cb = build_fullSize_Pic(Cb_PicPackage)
    Cr = build_fullSize_Pic(Cr_PicPackage)
    YUV = np.concatenate((np.expand_dims(Luma, 0), np.expand_dims(Cb, 0), np.expand_dims(Cr, 0)), 0)
    YUV = YUV.transpose((1, 2, 0))
    print(YUV.shape)
    print(YUV.dtype)
    img = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)
    print(img.shape)

    #cv2.imwrite('8.png', img)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)

    #cv2.imwrite('2.png', Luma)
    #cv2.imshow('img', Luma)
    #cv2.waitKey(0)

    Luma_PicPackage = r'D:\tashikete\debug_visualize\luma_32_32\CCLM'
    # Cb_PicPackage = r'D:\tashikete\debug_visualize\Cb_32_32\pred'
    # Cr_PicPackage = r'D:\tashikete\debug_visualize\Cr_32_32\pred'
    #Cb_PicPackage = r'D:\tashikete\debug_visualize\Cb_32_32\CCLM'
    #Cr_PicPackage = r'D:\tashikete\debug_visualize\Cr_32_32\CCLM'
    Cb_PicPackage = r'D:\tashikete\debug_visualize\Cb_32_32\org'
    Cr_PicPackage = r'D:\tashikete\debug_visualize\Cr_32_32\org'

    Luma = build_fullSize_Pic(Luma_PicPackage)

    Cb = build_fullSize_Pic(Cb_PicPackage)
    Cr = build_fullSize_Pic(Cr_PicPackage)
    YUV = np.concatenate((np.expand_dims(Luma, 0), np.expand_dims(Cb, 0), np.expand_dims(Cr, 0)), 0)
    YUV = YUV.transpose((1, 2, 0))
    print(YUV.shape)
    print(YUV.dtype)
    img2 = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)
    print(img2.shape)

    MSE_Loss = nn.MSELoss()
    #img = img.astype(np.double)
    #img2 = img2.astype(np.double)
    img = torch.FloatTensor(img);
    img2 = torch.FloatTensor(img2);
    loss = MSE_Loss(img, img2)
    print(loss)


    '''
    if not self.result_path.exists():
        self.result_path.mkdir()

    self.G.eval()

    # test_data_loader
    original_, sketch_, iv_tag_, cv_tag_ = self.test_images
    image_frame_dim = int(np.ceil(np.sqrt(len(original_))))

    # iv_tag_ to feature tensor 16 * 16 * 256 by pre-reained Sketch.
    with torch.no_grad():
        feature_tensor = self.Pretrain_ResNeXT(sketch_)

        if self.gpu_mode:
            original_, sketch_, iv_tag_, cv_tag_, feature_tensor = original_.to(self.device), sketch_.to \
                (self.device), iv_tag_.to(self.device), cv_tag_.to(self.device), feature_tensor.to(self.device)

        G_f, G_g = self.G(sketch_, feature_tensor, cv_tag_)

        if self.gpu_mode:
            G_f = G_f.cpu()
            G_g = G_g.cpu()

        G_f = self.color_revert(G_f)
        G_g = self.color_revert(G_g)

    utils.save_images(G_f[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      self.result_path / 'tag2pix_epoch{:03d}_G_f.png'.format(epoch))
    utils.save_images(G_g[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      self.result_path / 'tag2pix_epoch{:03d}_G_g.png'.format(epoch))
    '''

if __name__ == '__main__':
    visualize_results(0)
