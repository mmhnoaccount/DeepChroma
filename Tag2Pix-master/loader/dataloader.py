import pickle, random 
import math, time, platform
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from skimage import color

from torchvision import transforms, datasets
from torchvision.transforms import functional as tvF
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets

import os
from temp_visualize import sort_humanly

def set_seed(seed, print_log=True):
    if seed < 0:
        return
    if print_log:
        print('set random seed: {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
def pseudo_uniform(id, a, b):
    return (((id * 1.253 + a * 324.2351 + b * 534.342) * 20147.2312369804) + 0.12949) % (b - a) + a

def real_uniform(id, a, b):
    return random.uniform(a, b)

def get_tag_dict(tag_dump_path):
    with open(tag_dump_path, 'rb') as f:
        pkl = pickle.load(f)
        iv_tag_list = pkl['iv_tag_list']
        cv_tag_list = pkl['cv_tag_list']
        name_to_id =  pkl['tag_dict']

    iv_dict = {tag_id: i for (i, tag_id) in enumerate(iv_tag_list)}
    cv_dict = {tag_id: i for (i, tag_id) in enumerate(cv_tag_list)}
    id_to_name = {tag_id: tag_name for (tag_name, tag_id) in name_to_id.items()}

    return (iv_dict, cv_dict, id_to_name)

def read_tagline_txt(tag_txt_path, img_dir_path, iv_dict, cv_dict, data_size=0, is_train=True, seed=-1):
    iv_class_len = len(iv_dict)
    cv_class_len = len(cv_dict)
    print("read_tagline_txt! We will use %d, %d tags" % (iv_class_len, cv_class_len))

    if not tag_txt_path.exists():
        raise Exception(f'tag list text file "{tag_txt_path}" does not exist.')

    iv_tag_set = set(iv_dict.keys())
    cv_tag_set = set(cv_dict.keys())
    iv_class_list = []
    cv_class_list = []
    file_id_list = []

    data_limited = data_size != 0
    count = 0
    count_all = 0
    all_tag_num = 0
    awful_tag_num = 0
    iv_tag_num = 0
    cv_tag_num = 0

    include_tags = [470575, 540830] # 1girl, 1boy
    hair_tags = [87788, 16867, 13200, 10953, 16442, 11429, 15425, 8388, 5403, 16581, 87676, 16580, 94007, 403081, 468534]
    eye_tags = [10959, 8526, 16578, 10960, 15654, 89189, 16750, 13199, 89368, 95405, 89228, 390186]

    tag_lines = []
    with tag_txt_path.open('r') as f:
        for line in f:
            tag_lines.append(line)

    random.seed(10)
    random.shuffle(tag_lines)
    random.seed(time.time() if seed < 0 else seed)

    for line in tag_lines:
        count_all += 1
        tag_str_list = line.split(' ')
        tag_list = [int(i) for i in tag_str_list]

        file_name = tag_list[0]
        tag_list = set(tag_list[1:])

        if not (img_dir_path / f'{file_name}.png').exists():
            continue

        # one girl or one boy / one hair and eye color
        person_tag = tag_list.intersection(include_tags)
        hair_tag = tag_list.intersection(hair_tags)
        eye_tag = tag_list.intersection(eye_tags)

        if not (len(hair_tag) == 1 and len(eye_tag) == 1 and len(person_tag) == 1):
            awful_tag_num += 1
            if is_train:
                continue

        iv_class = torch.zeros(iv_class_len, dtype=torch.float)
        cv_class = torch.zeros(cv_class_len, dtype=torch.float)
        tag_exist = False

        for tag in tag_list:
            if tag in iv_tag_set:
                try:
                    iv_class[iv_dict[tag]] = 1
                    tag_exist = True
                    iv_tag_num += 1
                except IndexError as e:
                    print(len(iv_dict), iv_class_len, tag, iv_dict[tag])
                    raise e

        if not tag_exist and is_train:
            continue
        tag_exist = False

        for tag in tag_list:
            if tag in cv_tag_set:
                try:
                    cv_class[cv_dict[tag]] = 1
                    tag_exist = True
                    cv_tag_num += 1
                except IndexError as e:
                    print(len(cv_dict), cv_class_len, tag, cv_dict[tag])
                    raise e

        if not tag_exist and is_train:
            continue

        file_id_list.append(file_name)
        iv_class_list.append(iv_class)
        cv_class_list.append(cv_class)

        all_tag_num += len(tag_list)
        count += 1
        if data_limited and count > data_size:
            break

    print(f'count_all {count_all}, select_count {count}, awful_count {awful_tag_num}, all_tag_num {all_tag_num}, iv_tag_num {iv_tag_num}, cv_tag_num {cv_tag_num}')
    return (file_id_list, iv_class_list, cv_class_list)


class ColorAndSketchDataset(Dataset):
    def __init__(self, rgb_path, sketch_path_list, file_id_list, iv_class_list, cv_class_list,
            override_len=None, both_transform=None, sketch_transform=None, color_transform=None, seed=-1, **kwargs):

        self.rgb_path = rgb_path
        self.sketch_path_list = sketch_path_list

        self.file_id_list = file_id_list # copy

        self.iv_class_list = iv_class_list
        self.cv_class_list = cv_class_list

        self.both_transform = both_transform
        self.color_transform = color_transform
        self.sketch_transform = sketch_transform
        self.data_len = len(file_id_list)

        if override_len > 0 and self.data_len > override_len:
            self.data_len = override_len
        self.idx_shuffle = list(range(self.data_len))

        random.seed(10)
        random.shuffle(self.idx_shuffle)
        random.seed(time.time() if seed < 0 else seed)

    def __getitem__(self, idx):
        index = self.idx_shuffle[idx]

        file_id = self.file_id_list[index]
        iv_tag_class = self.iv_class_list[index]
        cv_tag_class = self.cv_class_list[index]

        sketch_path = random.choice(self.sketch_path_list)
        color_path = self.rgb_path / f"{file_id}.png"
        sketch_path = sketch_path / f"{file_id}.png"

        color_img = Image.open(color_path).convert('RGB')
        sketch_img = Image.open(sketch_path).convert('L')  # to [1, H, W]

        if self.both_transform is not None:
            color_img, sketch_img = self.both_transform(color_img, sketch_img)
        if self.color_transform is not None:
            color_img = self.color_transform(color_img)
        if self.sketch_transform is not None:
            sketch_img = self.sketch_transform(sketch_img)

        return (color_img, sketch_img, iv_tag_class, cv_tag_class)

    def __len__(self):
        return self.data_len

    def enhance_brightness(self, input_size):
        random_jitter = [transforms.ColorJitter(brightness=[1, 7], contrast=0.2, saturation=0.2)]
        data_augmentation = [transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS),
                            transforms.ToTensor()]
        self.sketch_transform = transforms.Compose(random_jitter + data_augmentation)

class RGB2ColorSpace(object):
    def __init__(self, color_space):
        self.color_space = color_space

    def __call__(self, img):
        if self.color_space == 'rgb':
            return (img * 2 - 1.)

        img = img.permute(1, 2, 0) # to [H, W, 3]
        if self.color_space == 'lab':
            img = color.rgb2lab(img) # [0~100, -128~127, -128~127]
            img[:,:,0] = (img[:,:,0] - 50.0) * (1 / 50.)
            img[:,:,1] = (img[:,:,1] + 0.5) * (1 / 127.5)
            img[:,:,2] = (img[:,:,2] + 0.5) * (1 / 127.5)
        elif self.color_space == 'hsv':
            img = color.rgb2hsv(img) # [0~1, 0~1, 0~1]
            img = (img * 2 - 1)

        # to [3, H, W]
        return torch.from_numpy(img).float().permute(2, 0, 1) # [-1~1, -1~1, -1~1]

class ColorSpace2RGB(object):
    """
    [-1, 1] to [0, 255]
    """
    def __init__(self, color_space):
        self.color_space = color_space

    def __call__(self, img):
        """numpy array [b, [-1~1], [-1~1], [-1~1]] to target space / result rgb[0~255]"""
        img = img.data.numpy()

        if self.color_space == 'rgb':
            img = (img + 1) * 0.5

        img = img.transpose(0, 2, 3, 1)
        if self.color_space == 'lab': # to [0~100, -128~127, -128~127]
            img[:,:,:,0] = (img[:,:,:,0] + 1) * 50
            img[:,:,:,1] = (img[:,:,:,1] * 127.5) - 0.5
            img[:,:,:,2] = (img[:,:,:,2] * 127.5) - 0.5
            img_list = []
            for i in img:
                img_list.append(color.lab2rgb(i))
            img = np.array(img_list)
        elif self.color_space == 'hsv': # to [0~1, 0~1, 0~1]
            img = (img + 1) * 0.5
            img_list = []
            for i in img:
                img_list.append(color.hsv2rgb(i))
            img = np.array(img_list)

        img = (img * 255).astype(np.uint8)
        return img # [0~255] / [b, h, w, 3]


def rot_crop(x):
    """return maximum width ratio of rotated image without letterbox"""
    x = abs(x)
    deg45 = math.pi * 0.25
    deg135 = math.pi * 0.75
    x = x * math.pi / 180
    a = (math.sin(deg135 - x) - math.sin(deg45 - x))/(math.cos(deg135-x)-math.cos(deg45-x))
    return math.sqrt(2) * (math.sin(deg45-x) - a*math.cos(deg45-x)) / (1-a)

class RandomFRC(transforms.RandomResizedCrop):
    """RandomHorizontalFlip + RandomRotation + RandomResizedCrop 2 images"""
    def __call__(self, img1, img2):
        img1 = tvF.resize(img1, self.size, interpolation=Image.LANCZOS)
        img2 = tvF.resize(img2, self.size, interpolation=Image.LANCZOS)
        if random.random() < 0.5:
            img1 = tvF.hflip(img1)
            img2 = tvF.hflip(img2)
        if random.random() < 0.5:
            rot = random.uniform(-10, 10)
            crop_ratio = rot_crop(rot)
            img1 = tvF.rotate(img1, rot, resample=Image.BILINEAR)
            img2 = tvF.rotate(img2, rot, resample=Image.BILINEAR)
            img1 = tvF.center_crop(img1, int(img1.size[0] * crop_ratio))
            img2 = tvF.center_crop(img2, int(img2.size[0] * crop_ratio))

        i, j, h, w = self.get_params(img1, self.scale, self.ratio)

        # return the image with the same transformation
        return (tvF.resized_crop(img1, i, j, h, w, self.size, self.interpolation),
                tvF.resized_crop(img2, i, j, h, w, self.size, self.interpolation))

def get_train_dataset(args):
    set_seed(args.seed)

    data_dir_path = Path(args.data_dir)

    batch_size = args.batch_size
    input_size = args.input_size

    data_randomize = RandomFRC(input_size, scale=(0.9, 1.0), ratio=(0.95, 1.05), interpolation=Image.LANCZOS)

    swap_color_space = [RGB2ColorSpace(args.color_space)]
    random_jitter = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    data_augmentation = [transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS),
                        transforms.ToTensor()]

    iv_dict, cv_dict, id_to_name = get_tag_dict(args.tag_dump)

    iv_class_len = len(iv_dict.keys())
    cv_class_len = len(cv_dict.keys())

    data_size = args.data_size
    tag_path = data_dir_path / args.tag_txt

    # Train set
    print('making train set...')

    rgb_train_path = data_dir_path / "rgb_train"
    sketch_dir_path_list = ["keras_train", "simpl_train", "xdog_train"]
    sketch_dir_path_list = [data_dir_path / p for p in sketch_dir_path_list if (data_dir_path / p).exists()]

    (train_id_list, train_iv_class_list, train_cv_class_list) = read_tagline_txt(
        tag_path, rgb_train_path, iv_dict, cv_dict, data_size=data_size, is_train=True, seed=args.seed)

    if platform.system() == 'Windows':
        _init_fn = None
    else:
        _init_fn = lambda worker_id: set_seed(args.seed, print_log=False)

    train = ColorAndSketchDataset(rgb_path=rgb_train_path, sketch_path_list=sketch_dir_path_list,
        file_id_list=train_id_list, iv_class_list=train_iv_class_list, cv_class_list=train_cv_class_list,
        override_len=data_size, both_transform=data_randomize,
        sketch_transform=transforms.Compose(random_jitter + data_augmentation),
        color_transform=transforms.Compose(data_augmentation + swap_color_space),
        seed=args.seed)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=args.thread, worker_init_fn=_init_fn)

    print(f'iv_class_len={iv_class_len}, cv_class_len={cv_class_len}')
    print(f'train: read {sketch_dir_path_list[0]}, id_list len={len(train_id_list)}, iv_class len={len(train_iv_class_list)}, cv_class len={len(train_cv_class_list)}')


    # Test set
    print('making test set...')

    rgb_test_path = data_dir_path / "benchmark"
    sketch_test_path = data_dir_path / "keras_test"

    (test_id_list, test_iv_class_list, test_cv_class_list) = read_tagline_txt(
        tag_path, rgb_test_path, iv_dict, cv_dict, is_train=False, data_size=args.test_image_count)

    test = ColorAndSketchDataset(rgb_path=rgb_test_path, sketch_path_list=[sketch_test_path],
        file_id_list=test_id_list, iv_class_list=test_iv_class_list, cv_class_list=test_cv_class_list,
        override_len=args.test_image_count, both_transform=data_randomize,
        sketch_transform=transforms.Compose(data_augmentation),
        color_transform=transforms.Compose(data_augmentation + swap_color_space))

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread, worker_init_fn=_init_fn)

    print(f'test: read {sketch_test_path}, id_list len={len(test_id_list)}, iv_class len={len(test_iv_class_list)}, cv_class len={len(test_cv_class_list)}')

    return train_loader, test_loader


class LinerTestDataset(Dataset):
    def __init__(self, sketch_path, file_id_list, iv_class_list, cv_class_list,
            override_len=None, sketch_transform=None, **kwargs):
        self.sketch_path = sketch_path

        self.file_id_list = file_id_list # copy

        self.iv_class_list = iv_class_list
        self.cv_class_list = cv_class_list

        self.sketch_transform = sketch_transform
        self.data_len = len(file_id_list)

        if override_len > 0 and self.data_len > override_len:
            self.data_len = override_len

    def __getitem__(self, idx):
        file_id = self.file_id_list[idx]

        iv_tag_class = self.iv_class_list[idx]
        cv_tag_class = self.cv_class_list[idx]

        sketch_path = self.sketch_path / f"{file_id}.png"

        sketch_img = Image.open(sketch_path).convert('L')  # to [1, H, W]
        if self.sketch_transform is not None:
            sketch_img = self.sketch_transform(sketch_img)

        return (sketch_img, file_id, iv_tag_class, cv_tag_class)

    def __len__(self):
        return self.data_len

def get_test_dataset(args):
    data_dir_path = Path(args.data_dir)

    batch_size = args.batch_size
    input_size = args.input_size

    data_augmentation = [transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS),
                        transforms.ToTensor()]

    iv_dict, cv_dict, _ = get_tag_dict(args.tag_dump)

    iv_class_len = len(iv_dict.keys())
    cv_class_len = len(cv_dict.keys())

    print('reading tagline')
    data_size = args.data_size

    sketch_path = data_dir_path / args.test_dir
    tag_path = data_dir_path / args.tag_txt

    (test_id_list, test_iv_class_list, test_cv_clas_list) = read_tagline_txt(
        tag_path, sketch_path, iv_dict, cv_dict, is_train=False, data_size=data_size)

    print('making train set...')

    test_dataset = LinerTestDataset(sketch_path=sketch_path, file_id_list=test_id_list, 
        iv_class_list=test_iv_class_list, cv_class_list=test_cv_clas_list,
        override_len=data_size, sketch_transform=transforms.Compose(data_augmentation))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.thread)

    print(f'iv_class_len={iv_class_len}, cv_class_len={cv_class_len}')

    return test_loader

def get_dataset(args):
    if args.test:
        return get_test_dataset(args)
    else:
        return get_train_dataset(args)

'''





TrainSet_Pack = 'F:/train_set'
Train_Cb_org = TrainSet_Pack + '/Cb_32_32/org'
Train_Cr_org = TrainSet_Pack + '/Cr_32_32/org'
Train_Luma = TrainSet_Pack + '/luma_32_32/CCLM'
Train_Cb_CCLM = TrainSet_Pack + '/Cb_32_32/CCLM'
Train_Cr_CCLM = TrainSet_Pack + '/Cr_32_32/CCLM'
Train_QP = TrainSet_Pack + '/QP_32_32'

Train_Cb_DM = TrainSet_Pack + '/Cb_32_32/DM'
Train_Cr_DM = TrainSet_Pack + '/Cr_32_32/DM'
Train_Cb_Ref = TrainSet_Pack + '/Cb_32_32/Ref'
Train_Cr_Ref = TrainSet_Pack + '/Cr_32_32/Ref'


TestSet_Pack = 'F:/test_set'
Test_Cb_org = TestSet_Pack + '/Cb_32_32/org'
Test_Cr_org = TestSet_Pack + '/Cr_32_32/org'
Test_Luma = TestSet_Pack + '/luma_32_32/CCLM'
Test_Cb_CCLM = TestSet_Pack + '/Cb_32_32/CCLM'
Test_Cr_CCLM = TestSet_Pack + '/Cr_32_32/CCLM'
Test_QP = TestSet_Pack + '/QP_32_32'

Test_Cb_DM = TestSet_Pack + '/Cb_32_32/DM'
Test_Cr_DM = TestSet_Pack + '/Cr_32_32/DM'
Test_Cb_Ref = TestSet_Pack + '/Cb_32_32/Ref'
Test_Cr_Ref = TestSet_Pack + '/Cr_32_32/Ref'


VisualSet_pack = 'E:/visualize_set'
Visualize_Cb_org = VisualSet_pack + '/train_visualize_22/Cb_32_32/org'
Visualize_Cr_org = VisualSet_pack + '/train_visualize_22/Cr_32_32/org'
Visualize_Luma = VisualSet_pack + '/train_visualize_22/luma_32_32/CCLM'
Visualize_Cb_CCLM = VisualSet_pack + '/train_visualize_22/Cb_32_32/CCLM'
Visualize_Cr_CCLM = VisualSet_pack + '/train_visualize_22/Cr_32_32/CCLM'
Visualize_QP = VisualSet_pack + '/train_visualize_22/QP_32_32'

Visualize_Cb_DM = VisualSet_pack + '/train_visualize_22/Cb_32_32/DM'
Visualize_Cr_DM = VisualSet_pack + '/train_visualize_22/Cr_32_32/DM'
Visualize_Cb_Ref = VisualSet_pack + '/train_visualize_22/Cb_32_32/Ref'
Visualize_Cr_Ref = VisualSet_pack + '/train_visualize_22/Cr_32_32/Ref'

def readMatrix(path, QP_mask=False):
    height = 32
    width = 32
    A = np.zeros((height, width), dtype=float)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型

    f = open(path)  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        per_line = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
        # print(list)
        A[A_row, :] = per_line[:-1]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
        # print(line)
    if QP_mask:
        A = (A-(63.0/2.0)) / (63.0/2.0)
    else:
        A = (A-(1023.0/2.0)) / (1023.0/2.0)
    return A

def readRef(path):
    height = 32
    width = 32
    A = np.zeros((1, height + width), dtype=float)
    f = open(path)  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    line = lines[0]
    per_line = line.strip('\n').split(' ')
    #print(per_line)
    #print(per_line.shape)
    A[0, :] = per_line[:-1]
    A = (A-(1023.0/2.0)) / (1023.0/2.0)
    return A


def catLuma(luma_path, index, Train, Visualize=False):
    if Visualize:
        Luma_Matrix = readMatrix(Visualize_Luma + '/' + luma_path[0][index])
        Cb_CCLM_Matrix = readMatrix(Visualize_Cb_CCLM + '/' + luma_path[1][index])
        Cr_CCLM_Matrix = readMatrix(Visualize_Cr_CCLM + '/' + luma_path[2][index])
        QP_Matrix = readMatrix(Visualize_QP + '/' + luma_path[3][index], QP_mask=True)
    else:
        if Train:
            Luma_Matrix = readMatrix(Train_Luma + '/' + luma_path[0][index])
            Cb_CCLM_Matrix = readMatrix(Train_Cb_CCLM + '/' + luma_path[1][index])
            Cr_CCLM_Matrix = readMatrix(Train_Cr_CCLM + '/' + luma_path[2][index])
            QP_Matrix = readMatrix(Train_QP + '/' + luma_path[3][index], QP_mask=True)
        else:
            Luma_Matrix = readMatrix(Test_Luma + '/' + luma_path[0][index])
            Cb_CCLM_Matrix = readMatrix(Test_Cb_CCLM + '/' + luma_path[1][index])
            Cr_CCLM_Matrix = readMatrix(Test_Cr_CCLM + '/' + luma_path[2][index])
            QP_Matrix = readMatrix(Test_QP + '/' + luma_path[3][index], QP_mask=True)
    return np.concatenate((np.expand_dims(Luma_Matrix, 0),
                           np.expand_dims(Cb_CCLM_Matrix, 0),
                           np.expand_dims(Cr_CCLM_Matrix, 0),
                           np.expand_dims(QP_Matrix, 0)), 0)

def catChroma(chroma_path, index, Train, Visualize=False):
    if Visualize:
        Cb_DM_Matrix = readMatrix(Visualize_Cb_DM + '/' + chroma_path[0][index])
        Cr_DM_Matrix = readMatrix(Visualize_Cr_DM + '/' + chroma_path[1][index])
    else:
        if Train:
            Cb_DM_Matrix = readMatrix(Train_Cb_DM + '/' + chroma_path[0][index])
            Cr_DM_Matrix = readMatrix(Train_Cr_DM + '/' + chroma_path[1][index])
        else:
            Cb_DM_Matrix = readMatrix(Test_Cb_DM + '/' + chroma_path[0][index])
            Cr_DM_Matrix = readMatrix(Test_Cr_DM + '/' + chroma_path[1][index])
    return np.concatenate((np.expand_dims(Cb_DM_Matrix, 0), np.expand_dims(Cr_DM_Matrix, 0)), 0)

def catOrg(org_path, index, Train, Visualize=False):
    if Visualize:
        Cb_org_Matrix = readMatrix(Visualize_Cb_org + '/' + org_path[0][index])
        Cr_org_Matrix = readMatrix(Visualize_Cr_org + '/' + org_path[1][index])
    else:
        if Train:
            Cb_org_Matrix = readMatrix(Train_Cb_org + '/' + org_path[0][index])
            Cr_org_Matrix = readMatrix(Train_Cr_org + '/' + org_path[1][index])
        else:
            Cb_org_Matrix = readMatrix(Test_Cb_org + '/' + org_path[0][index])
            Cr_org_Matrix = readMatrix(Test_Cr_org + '/' + org_path[1][index])
    return np.concatenate((np.expand_dims(Cb_org_Matrix, 0), np.expand_dims(Cr_org_Matrix, 0)), 0)

def catRef(Ref_path, index, Train, Visualize=False):
    if Visualize:
        Cb_Ref_Matrix = readRef(Visualize_Cb_Ref + '/' + Ref_path[2][index])
        Cr_Ref_Matrix = readRef(Visualize_Cr_Ref + '/' + Ref_path[3][index])
    else:
        if Train:
            Cb_Ref_Matrix = readRef(Train_Cb_Ref + '/' + Ref_path[2][index])
            Cr_Ref_Matrix = readRef(Train_Cr_Ref + '/' + Ref_path[3][index])
        else:
            Cb_Ref_Matrix = readRef(Test_Cb_Ref + '/' + Ref_path[2][index])
            Cr_Ref_Matrix = readRef(Test_Cr_Ref + '/' + Ref_path[3][index])
    return np.concatenate((Cb_Ref_Matrix, Cr_Ref_Matrix), 1)


class LumaAndChromaDataset(Dataset):
    def __init__(self, luma_path, chroma_path, org_path, Train=False, seed=-1, **kwargs):

        self.luma_path = luma_path
        self.chroma_path = chroma_path
        self.org_path = org_path
        self.Train = Train

        self.data_len = len(luma_path[0])

        self.idx_shuffle = list(range(self.data_len))

        random.seed(10)
        random.shuffle(self.idx_shuffle)
        random.seed(time.time() if seed < 0 else seed)

    def __getitem__(self, idx):
        index = self.idx_shuffle[idx]

        Luma_id = torch.FloatTensor(catLuma(self.luma_path, index, self.Train))
        Chorma_id = torch.FloatTensor(catChroma(self.chroma_path, index, self.Train))
        Ref_id = torch.FloatTensor(catRef(self.chroma_path, index, self.Train)).squeeze()

        org_id = torch.FloatTensor(catOrg(self.org_path, index, self.Train))

        return (Luma_id, Chorma_id, Ref_id, org_id)

    def __len__(self):
        return self.data_len


class visualize_Dataset(Dataset):
    def __init__(self):
        Luma_list = sort_humanly(os.listdir(Visualize_Luma))
        Cb_CCLM_list = sort_humanly(os.listdir(Visualize_Cb_CCLM))
        Cr_CCLM_list = sort_humanly(os.listdir(Visualize_Cr_CCLM))
        QP_list = sort_humanly(os.listdir(Visualize_QP))

        Cb_DM_list = sort_humanly(os.listdir(Visualize_Cb_DM))
        Cr_DM_list = sort_humanly(os.listdir(Visualize_Cr_DM))
        Cb_Ref_list = sort_humanly(os.listdir(Visualize_Cb_Ref))
        Cr_Ref_list = sort_humanly(os.listdir(Visualize_Cr_Ref))

        Cb_org_list = sort_humanly(os.listdir(Visualize_Cb_org))
        Cr_org_list = sort_humanly(os.listdir(Visualize_Cr_org))

        Train_len = len(Luma_list)

        Luma_path = (Luma_list, Cb_CCLM_list, Cr_CCLM_list, QP_list)
        Chroma_path = (Cb_DM_list, Cr_DM_list, Cb_Ref_list, Cr_Ref_list)
        org_path = (Cb_org_list, Cr_org_list)


        self.luma_path = Luma_path
        self.chroma_path = Chroma_path
        self.org_path = org_path
        self.Train = False
        self.Visualize = True

        self.data_len = len(Luma_path[0])

    def __getitem__(self, idx):
        index = idx

        Luma_id = torch.FloatTensor(catLuma(self.luma_path, index, self.Train, self.Visualize))
        Chorma_id = torch.FloatTensor(catChroma(self.chroma_path, index, self.Train, self.Visualize))
        Ref_id = torch.FloatTensor(catRef(self.chroma_path, index, self.Train, self.Visualize)).squeeze()

        org_id = torch.FloatTensor(catOrg(self.org_path, index, self.Train, self.Visualize))

        return (Luma_id, Chorma_id, Ref_id, org_id)

    def __len__(self):
        return self.data_len



def get_train_dataset(args):
    set_seed(args.seed)

    #data_dir_path = Path(args.data_dir)

    batch_size = args.batch_size
    input_size = args.input_size

    Train_Luma_list = os.listdir(Train_Luma)
    Train_Cb_CCLM_list = os.listdir(Train_Cb_CCLM)
    Train_Cr_CCLM_list = os.listdir(Train_Cr_CCLM)
    Train_QP_list = os.listdir(Train_QP)

    Train_Cb_DM_list = os.listdir(Train_Cb_DM)
    Train_Cr_DM_list = os.listdir(Train_Cr_DM)
    Train_Cb_Ref_list = os.listdir(Train_Cb_Ref)
    Train_Cr_Ref_list = os.listdir(Train_Cr_Ref)

    Train_Cb_org_list = os.listdir(Train_Cb_org)
    Train_Cr_org_list = os.listdir(Train_Cr_org)

    Train_len = len(Train_Luma_list)

    Train_Luma_path = (Train_Luma_list, Train_Cb_CCLM_list, Train_Cr_CCLM_list, Train_QP_list)
    Train_Chroma_path = (Train_Cb_DM_list, Train_Cr_DM_list, Train_Cb_Ref_list, Train_Cr_Ref_list)
    Train_org_path = (Train_Cb_org_list, Train_Cr_org_list)

    # Train set
    print('making train set...')

    if platform.system() == 'Windows':
        _init_fn = None
    else:
        _init_fn = lambda worker_id: set_seed(args.seed, print_log=False)

    train = LumaAndChromaDataset(luma_path=Train_Luma_path, chroma_path=Train_Chroma_path,
        org_path=Train_org_path, Train=True, seed=args.seed)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=args.thread, worker_init_fn=_init_fn)


    # Test set
    print('making test set...')

    Test_Luma_list = os.listdir(Test_Luma)
    Test_Cb_CCLM_list = os.listdir(Test_Cb_CCLM)
    Test_Cr_CCLM_list = os.listdir(Test_Cr_CCLM)
    Test_QP_list = os.listdir(Test_QP)

    Test_Cb_DM_list = os.listdir(Test_Cb_DM)
    Test_Cr_DM_list = os.listdir(Test_Cr_DM)
    Test_Cb_Ref_list = os.listdir(Test_Cb_Ref)
    Test_Cr_Ref_list = os.listdir(Test_Cr_Ref)

    Test_Cb_org_list = os.listdir(Test_Cb_org)
    Test_Cr_org_list = os.listdir(Test_Cr_org)

    Test_len = len(Test_Luma_list)

    Test_Luma_path = (Test_Luma_list, Test_Cb_CCLM_list, Test_Cr_CCLM_list, Test_QP_list)
    Test_Chroma_path = (Test_Cb_DM_list, Test_Cr_DM_list, Test_Cb_Ref_list, Test_Cr_Ref_list)
    Test_org_path = (Test_Cb_org_list, Test_Cr_org_list)

    test = LumaAndChromaDataset(luma_path=Test_Luma_path, chroma_path=Test_Chroma_path,
                                 org_path=Test_org_path, Train=False)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread,
                              worker_init_fn=_init_fn)

    return train_loader, test_loader


def get_test_dataset(args):

    batch_size = args.batch_size
    input_size = args.input_size

    # Test set
    print('making test set...')

    Test_Luma_list = os.listdir(Test_Luma)
    Test_Cb_CCLM_list = os.listdir(Test_Cb_CCLM)
    Test_Cr_CCLM_list = os.listdir(Test_Cr_CCLM)
    Test_QP_list = os.listdir(Test_QP)

    Test_Cb_DM_list = os.listdir(Test_Cb_DM)
    Test_Cr_DM_list = os.listdir(Test_Cr_DM)
    Test_Cb_Ref_list = os.listdir(Test_Cb_Ref)
    Test_Cr_Ref_list = os.listdir(Test_Cr_Ref)

    Test_Cb_org_list = os.listdir(Test_Cb_org)
    Test_Cr_org_list = os.listdir(Test_Cr_org)

    Test_len = len(Test_Luma_list)

    Test_Luma_path = (Test_Luma_list, Test_Cb_CCLM_list, Test_Cr_CCLM_list, Test_QP_list)
    Test_Chroma_path = (Test_Cb_DM_list, Test_Cr_DM_list, Test_Cb_Ref_list, Test_Cr_Ref_list)
    Test_org_path = (Test_Cb_org_list, Test_Cr_org_list)

    test = LumaAndChromaDataset(luma_path=Test_Luma_path, chroma_path=Test_Chroma_path,
                                org_path=Test_org_path, Train=False)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=args.thread)

    return test_loader



def get_dataset(args):
    if args.test:
        return get_test_dataset(args)
    else:
        return get_train_dataset(args)


if __name__ == '__main__':
    batch_size = 1

    Test_Luma_list = os.listdir(Test_Luma)
    Test_Cb_CCLM_list = os.listdir(Test_Cb_CCLM)
    Test_Cr_CCLM_list = os.listdir(Test_Cr_CCLM)
    Test_QP_list = os.listdir(Test_QP)

    Test_Cb_DM_list = os.listdir(Test_Cb_DM)
    Test_Cr_DM_list = os.listdir(Test_Cr_DM)
    Test_Cb_Ref_list = os.listdir(Test_Cb_Ref)
    Test_Cr_Ref_list = os.listdir(Test_Cr_Ref)

    Test_Cb_org_list = os.listdir(Test_Cb_org)
    Test_Cr_org_list = os.listdir(Test_Cr_org)

    Test_Luma_path = (Test_Luma_list, Test_Cb_CCLM_list, Test_Cr_CCLM_list, Test_QP_list)
    Test_Chroma_path = (Test_Cb_DM_list, Test_Cr_DM_list, Test_Cb_Ref_list, Test_Cr_Ref_list)
    Test_org_path = (Test_Cb_org_list, Test_Cr_org_list)

    test = LumaAndChromaDataset(luma_path=Test_Luma_path, chroma_path=Test_Chroma_path,
                                org_path=Test_org_path, Train=False)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    Luma_id, Chorma_id, Ref_id, org_id = next(iter(test_loader))
    print(Luma_id.shape, Chorma_id.shape, Ref_id.shape, org_id.shape)
    #print(Luma_id[0, 3, :, :])
