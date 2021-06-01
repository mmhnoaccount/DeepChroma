import torch
import torchvision
from network import Generator
import torch.nn as nn


# An instance of your model.
#model = lstm_model()
#model.load_state_dict(torch.load('ccc_model_50.pkl'))
net_opt = {
            # 'guide': True,
            'guide': False,     #noguide
            'relu': False,
            # 'relu': True,
            'bn': False,        #nobn
            #'bn': True,
        }
layers = [1, 1, 1, 1]
#layers = [6, 4, 3, 3]
#layers = [12, 8, 5, 5]
G = Generator(luma_size=32, chroma_size=32, ref_length=128,
                           luma_dim=4, output_dim=2, layers=layers, net_opt=net_opt)
G = nn.DataParallel(G)
#checkpoint = torch.load(str('C:/Users/PC5/Desktop/Tag2Pix-master/Tag2Pix-master/results/no_bn/210108-152753/tag2pix_50_epoch.pkl'))
#checkpoint = torch.load(str('E:/results/210203-134300/tag2pix_100_epoch.pkl'))
#checkpoint = torch.load(str('E:/results/210320-141735/tag2pix_60_epoch.pkl'))
#checkpoint = torch.load(str('E:/results/210327-020718_bn/tag2pix_100_epoch.pkl'))   #bn 效果超差
#checkpoint = torch.load(str('E:/results/210330-100744_bn_noguide/tag2pix_100_epoch.pkl'))   #bn noguide
#checkpoint = torch.load(str('E:/results/210401-222201_nobn_noguide/tag2pix_100_epoch.pkl'))  #nobn noguide
#checkpoint = torch.load(str('E:/results/210407-211232_nobn_onlyguide_12000/tag2pix_50_epoch.pkl'))  #nobn onlyguide 12000
#checkpoint = torch.load(str('E:/results/210410-161324_nobn_noguide_use_relu_15000/tag2pix_80_epoch.pkl'))  #nobn noguide use_relu 15000
#checkpoint = torch.load(str('E:/results/210413-135210_nobn_noguide_use_relu_60000/tag2pix_80_epoch.pkl'))  #nobn noguide use_relu 60000
#checkpoint = torch.load(str('E:/results/210418-185612_deepLayer__nobn_noguide_use_relu_60000/tag2pix_80_epoch.pkl'))  #nobn noguide use_relu 60000 deepLayer
#checkpoint = torch.load(str('E:/results/210507-163421_orig_useRelu_noBn_3000_per10/tag2pix_40_epoch.pkl'))  #orig_useRelu_noBn_3000_per10
checkpoint = torch.load(str('E:/results/210517-222302_orig_noBn_noGuide_12000_per10/tag2pix_50_epoch.pkl'))  #210517-222302_orig_noBn_noGuide_12000_per10
G.load_state_dict(checkpoint['G'])
G = G.module
device = torch.device("cpu")
G = G.to(device)



# An example input you would normally provide to your model's forward() method.
#example_in = torch.ones([1, 1, 1])
#example_h0 = (torch.zeros([1, 1, 1]), torch.zeros([1, 1, 1]))
#print(example_h0.shape)
example_Luma = torch.ones([1, 4, 32, 32]) * 0.6
example_Chroma = torch.ones([1, 2, 32, 32]) * 0.3
example_Ref = torch.ones([1, 128]) * 0.7

'''
G.eval()
G_f, G_ = G(example_Luma, example_Chroma, example_Ref)
print(G_f[0][0][5][5])
'''

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
#traced_script_module = torch.jit.trace(model, (example_in, example_h0))
#traced_script_module.save("Script_ccc_model_50.pt")
traced_script_module = torch.jit.trace(G, (example_Luma, example_Chroma, example_Ref))
#traced_script_module.save("Script_model_no_Bn_epoch50.pt")
#traced_script_module.save("E:/results/210203-134300/tag2pix_100_epoch.pt")
#traced_script_module.save("E:/results/210320-141735/tag2pix_60_epoch.pt")
#traced_script_module.save("E:/results/210327-020718_bn/tag2pix_100_epoch.pt")   #bn 效果超差
#traced_script_module.save("E:/results/210330-100744_bn_noguide/tag2pix_100_epoch.pt")   #bn noguide
#traced_script_module.save("E:/results/210401-222201_nobn_noguide/tag2pix_100_epoch.pt")   #nobn noguide
#traced_script_module.save("E:/results/210407-211232_nobn_onlyguide_12000/tag2pix_50_epoch.pt")  #nobn onlyguide 12000
#traced_script_module.save("E:/results/210410-161324_nobn_noguide_use_relu_15000/tag2pix_80_epoch.pt")  #nobn noguide use_relu 15000
#traced_script_module.save("E:/results/210413-135210_nobn_noguide_use_relu_60000/tag2pix_80_epoch.pt")  #nobn noguide use_relu 60000
#traced_script_module.save("E:/results/210418-185612_deepLayer__nobn_noguide_use_relu_60000/tag2pix_80_epoch.pt")  #nobn noguide use_relu 60000 deepLayer
#traced_script_module.save("E:/results/210507-163421_orig_useRelu_noBn_3000_per10/tag2pix_40_epoch.pt")  #nobn noguide use_relu 60000 deepLayer

# traced_script_module.save("E:/results/210517-222302_orig_noBn_noGuide_12000_per10/tag2pix_50_epoch.pt")  #210517-222302_orig_noBn_noGuide_12000_per10

out, _ = traced_script_module(example_Luma, example_Chroma, example_Ref)
print(out[0][0][5][5])

