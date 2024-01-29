# -*-coding:utf-8-*-
import torch
from tqdm import tqdm
import numpy as np
import logging
import copy
from torch.optim import lr_scheduler
import h5py
import torch.nn as nn
from resnet_18 import Resnet18
from MCSiT_MoE import ViT
from resnet_50 import resnet50
import csv
def train(model, criterion, optimizer, num_epochs,iter):
    batch = 1
    num = 1
    loss_weights = torch.ones((4, 1)).to(device="cuda:0")
    avg_task_loss = np.zeros((200, 4))
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    model.train()  # Set model to training mode
    running_loss = 0;running_correct1 = 0;now_epoch_num = 0;batch = 1;running_correct2 = 0;running_correct3 = 0;running_correct4 = 0;batch = 1;correct2=0;correct3=0;correct4=0;correct1=0
    pbar = tqdm(data_loader1,desc='Train')
    for input,label1,label2,label3,label4 in pbar:
        data1=input.cuda().float();label3 = label3.cuda().long();label2 = label2.cuda().long();label1 = label1.cuda().long();label4 = label4.cuda().long()
        model.zero_grad()
        output1,output2,output3,output4 = model(data1)
        _,pred1=torch.max(output1, 1); _,pred2=torch.max(output2, 1); _,pred3=torch.max(output3, 1); _,pred4=torch.max(output4, 1)
        loss1 = criterion(output1, label1)
        loss2 = criterion(output2, label2)
        loss3 = criterion(output3, label3)
        loss4 = criterion(output4, label4)
        if iter > 0 and num == 1:
            loss1_init = torch.tensor(loss1.detach().item());
            loss2_init = torch.tensor(loss2.detach().item());
            loss3_init = torch.tensor(loss3.detach().item());
            loss4_init = torch.tensor(loss4.detach().item())
            loss1=loss1/loss1_init;loss2=loss2/loss2_init;loss3=loss3/loss3_init;loss4=loss4/loss4_init;loss1=loss1/loss1_init
            num=num+1

        if iter > 0:
            loss = torch.tensor((loss1.detach().item(),loss2.detach().item(), loss3.detach().item(), loss4.detach().item()))
            avg_task_loss[batch - 1, :] = loss
            correct1 += (label1==pred1).float().mean().item()
            correct2 += (label2==pred2).float().mean().item()
            correct3 += (label3==pred3).float().mean().item()
            correct4 += (label4==pred4).float().mean().item()
            batch += 1

        if iter > 0 and batch == 200:
            rate_mean = []
            for i in range(4):
                rate_mean.append(np.mean(avg_task_loss[:,i]))
            KPI = [correct1 / (batch-1),correct2 / (batch-1)-0.3, correct3 / (batch-1), correct4 / (batch-1)]
        #
            for i in range(4):
                loss_weights[i] =  np.exp(rate_mean[i]/KPI[i]/4)
            correct1 = 0
            correct2 = 0
            correct3 = 0
            correct4 = 0
            batch = 1
            with open("./result_acc_loss.csv", 'a', newline='') as t1:
                writer_train1 = csv.writer(t1)
                writer_train1.writerow([loss_weights[0],loss_weights[1],loss_weights[2],loss_weights[3],KPI[0],KPI[1],KPI[2],rate_mean[0],rate_mean[1],rate_mean[2]])
        mean_we = loss_weights.mean()

        loss=loss_weights[0]/mean_we*loss1 + loss_weights[1]/mean_we*loss2 + loss_weights[2]/mean_we*loss3+ loss_weights[3]/mean_we*loss4
        loss.backward()
        optimizer.step()
            # statistics
        running_loss += loss.item() * data1.size(0)
        running_correct1 += pred1.eq(label1.view_as(pred1)).sum().item()
        running_correct2 += pred2.eq(label2.view_as(pred2)).sum().item()
        running_correct3 += pred3.eq(label3.view_as(pred3)).sum().item()
        running_correct4 += pred4.eq(label4.view_as(pred4)).sum().item()
        now_epoch_num += data1.size(0)
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
    return model

def Test(model, criterion):
    model.eval()  # Set model to training mode
    running_loss = 0;running_correct1 = 0;now_epoch_num = 0;batch = 1;running_correct2 = 0;running_correct3 = 0;running_correct4 = 0
    pbar = tqdm(data_loader_test, desc='Test')
    for data3, label1,label2,label3,label4 in pbar:
        data3 = data3.cuda().float();label3 = label3.cuda().long();label2 = label2.cuda().long();label1 = label1.cuda().long();label4 = label4.cuda().long()
        model.zero_grad()
        output1, output2, output3, output4 = model(data3)
        _, pred1 = torch.max(output1, 1);_, pred2 = torch.max(output2, 1);_, pred3 = torch.max(output3, 1);_, pred4 = torch.max(output4, 1)
        loss1 = criterion(output1, label1)
        loss2 = criterion(output2, label2)
        loss3 = criterion(output3, label3)
        loss4 = criterion(output4, label4)
        loss = 1.0*loss1 + 1.0*loss2 + 1.0*loss3+ 1.0*loss4
        # statistics
        running_loss += loss.item() * data3.size(0)
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
    return epoch_loss

if __name__ =='__main__':
    trainset = h5py.File(r"train_tasks_v2.mat")['XTrainIQ']
    trainset = np.transpose(trainset)
    trainset = torch.from_numpy(trainset)
    trainset1 = trainset.permute(3, 2, 1, 0)
    print(trainset1.size())

    test_data = h5py.File(r"test_tasks_v2.mat")
    x_test = test_data['XTrainIQ']
    x_test = np.transpose(x_test)
    testset = torch.from_numpy(x_test)
    testset = testset.permute(3, 2, 1, 0)
    label_modulation = np.arange(0, 12)
    y_train = label_modulation.repeat(20)
    y_train = np.tile(y_train, 21)
    y_train = np.tile(y_train, 10)
    y_train = np.tile(y_train, 5)
    y_train1 = torch.FloatTensor(y_train)
    y_test = label_modulation.repeat(10)
    y_test = np.tile(y_test, 21)
    y_test = np.tile(y_test, 10)
    y_test1 = np.tile(y_test, 5)
    y_test1 = torch.FloatTensor(y_test1)

    label_snr = np.arange(0, 21)
    y_train2 = label_snr.repeat(240)
    y_train2 = np.tile(y_train2, 10)
    y_train2 = np.tile(y_train2, 5)
    y_train2 = torch.FloatTensor(y_train2)
    y_test2 = label_snr.repeat(120)
    y_test2 = np.tile(y_test2, 10)
    y_test2 = np.tile(y_test2, 5)
    y_test2 = torch.FloatTensor(y_test2)

    label_fre = np.arange(0, 10)
    y_train3 = label_fre.repeat(5040)
    y_train3 = np.tile(y_train3, 5)
    y_train3 = torch.FloatTensor(y_train3)
    y_test3 = label_fre.repeat(21 * 120)
    y_test3 = np.tile(y_test3, 5)
    y_test3 = torch.FloatTensor(y_test3)

    label_sps = np.arange(0, 5)
    y_train4 = label_sps.repeat(50400)
    y_train4 = torch.FloatTensor(y_train4)
    y_test4 = label_sps.repeat(210 * 120)
    y_test4 = torch.FloatTensor(y_test4)
    dataset1 = torch.utils.data.TensorDataset(trainset1,y_train1,y_train2,y_train3,y_train4)
    dataset_test = torch.utils.data.TensorDataset(testset, y_test1,y_test2,y_test3,y_test4)

    data_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=128,shuffle=True, num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=True, num_workers=4)
    for i in range(0, 5):
        model = ViT()
        # model = Resnet18(10).to('cuda:0')
        # model=resnet50(10)
        model = model.to(device="cuda:0")
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()
        best_loss = 100
        for epoch in range(30):
            train(model, criterion, optimizer_ft, num_epochs=30,iter=epoch)
            Test_loss = Test(model, criterion)
            if Test_loss < best_loss:
                best_loss = Test_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存当前的训练集精度、测试集精度和最高测试集精度
                torch.save(best_model_wts, f'./Transformer_MCSiT_v2_top1_NDW_{i}.pth')
