import torch
from tqdm import tqdm
import numpy as np
import logging
import copy
from torch.optim import lr_scheduler
import h5py
import torch.nn.functional as F
import torch.nn as nn
from class_model.MCSiT_v2copy import ViT
import scipy.io as sio
from class_model.MCSiT_v2 import ViT
from class_model.resnet_18 import Resnet18
from class_model.resnet_50 import resnet50
from class_model.mobilenet import mobilenet
from class_model.shufflenet import shufflenet
def test(snr_test):
    n = 0
    result_loss=np.zeros((21,4))
    for i in range(-20,22, 2):

        test_X = testset[np.where(snr_test == i)[0]]
        test_Y1 = y_test1[np.where(snr_test == i)[0]]
        test_Y2 = y_test2[np.where(snr_test == i)[0]]
        test_Y3 = y_test3[np.where(snr_test == i)[0]]
        test_Y4 = y_test4[np.where(snr_test == i)[0]]

        test_X=torch.FloatTensor(test_X)
        test_Y1 = torch.FloatTensor(test_Y1)
        test_Y2 = torch.FloatTensor(test_Y2)
        test_Y3 = torch.FloatTensor(test_Y3)
        test_Y4 = torch.FloatTensor(test_Y4)
        dataset1 = torch.utils.data.TensorDataset(test_X, test_Y1, test_Y2, test_Y3, test_Y4)
        data_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

        model.eval()
        with torch.no_grad():
            running_loss = 0;running_correct1 = 0;now_epoch_num = 0;running_loss2 = 0;running_loss3 = 0;running_loss4 = 0;running_correct2=0;running_correct3=0;running_correct4=0
            running_value2 = 0
            pbar = tqdm(data_loader1, desc='Test')

            for data3, label1, label2, label3, label4 in pbar:
                data3 = data3.cuda().float();label3 = label3.cuda().unsqueeze(-1);label2 = label2.cuda().unsqueeze(-1);label1 = label1.cuda().long();label4 = label4.cuda().unsqueeze(-1)
                model.zero_grad()
                output1, output2, output3, output4 = model(data3)
                _, pred1 = torch.max(output1, 1)
                _, pred2 = torch.max(output2, 1)
                _, pred3 = torch.max(output3, 1)
                _, pred4 = torch.max(output4, 1)

                running_correct1 += pred1.eq(label1.view_as(pred1)).sum().item()
                running_correct2 += pred2.eq(label2.view_as(pred2)).sum().item()
                running_correct3 += pred3.eq(label3.view_as(pred3)).sum().item()
                running_correct4 += pred4.eq(label4.view_as(pred4)).sum().item()

                now_epoch_num += data3.size(0)


                # 在进度条的右边实时显示数据集类型、loss值和精度
                epoch_loss = running_loss / now_epoch_num
                epoch_acc1 = running_correct1 / now_epoch_num
                epoch_acc2 = running_correct2 / now_epoch_num
                epoch_acc3 = running_correct3 / now_epoch_num
                epoch_acc4 = running_correct4 / now_epoch_num
                pbar.set_postfix({
                    'Loss': '{:.4f}'.format(epoch_loss),
                    'Acc1': '{:.4f}'.format(epoch_acc1),
                    'Acc2': '{:.4f}'.format(epoch_acc2),
                    'Acc3': '{:.4f}'.format(epoch_acc3),
                    'Acc4': '{:.4f}'.format(epoch_acc4),
                })
                result_loss[n,:]=[epoch_acc1,epoch_acc2,epoch_acc3,epoch_acc4]
                torch.cuda.empty_cache()
            n=n+1
    return result_loss

if __name__ =='__main__':
    test_data = h5py.File(r"test_tasks_v2.mat")
    x_test = test_data['XTrainIQ']
    x_test = np.transpose(x_test)
    testset = np.transpose(x_test, (3, 2, 1, 0))
    label_modulation = np.arange(0, 12)
    y_test = label_modulation.repeat(10)
    y_test = np.tile(y_test, 21)
    y_test = np.tile(y_test, 10)
    y_test1 = np.tile(y_test, 5)


    label_snr = np.arange(0, 21)
    y_test2 = label_snr.repeat(120)
    y_test2 = np.tile(y_test2, 10)
    y_test2 = np.tile(y_test2, 5)

    label_fre = np.arange(0, 10)
    y_test3 = label_fre.repeat(21 * 120)
    y_test3 = np.tile(y_test3, 5)

    label_sps = np.arange(0, 5)
    y_test4 = label_sps.repeat(210 * 120)

    label_snr = np.arange(-20, 22, 2)
    snr_test = label_snr.repeat(120)
    snr_test = np.tile(snr_test, 10)
    snr_test = np.tile(snr_test, 5)
    for i in range(0,3):
        model = ViT()
        # model=Resnet18(10)
        # model=shufflenet()
        model = model.to(device="cuda:0")
        model.load_state_dict(torch.load(f'Transformer_mcsit_top2_24experts_NDW_{i}.pth',map_location='cuda:0'))

        Test_loss = test(snr_test)
        sio.savemat(f'mcsit_top2_24experts_NDW_{i}.mat',{'acc':np.array(Test_loss)})
        print(Test_loss)
