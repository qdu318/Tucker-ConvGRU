import os
import time
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable

from Function.function import MAE, RMSE, get_cores, update_Us, update_cores, get_fold_tensor
import numpy as np
import h5py
import tensorly as tl
from ConvGRU import ConvGRU
from Function.preprocess import process_IF


if __name__ == '__main__':

    # detect if CUDA is available or not
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor  # computation in GPU
        device = torch.device("cuda")
        print("cuda yes")
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    # data = np.load("data/Taxi-NYC.npz")["volume"][:, :, :, 1]
    # data = data.reshape(-1, 10, 20)
    # datasets=["Taxi-NYC"]

    # Manhattan = np.load("data/Taxi-Manhattan.npy")[-5856:,:,:]
    # data = Manhattan.reshape(-1, 15, 5)

    datasets=["TaxiBJ13","TaxiBJ14","TaxiBJ15","TaxiBJ16"]
    # datasets=["TaxiBJ13"]
    # Rs_list=[[5,5],[7,7],[9,9],[11,11],[13,13]]
    Rs_list=[[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]
    for dataset_name in datasets:
        for rs in Rs_list:
            t = time.time()
            filename = "data/"+ dataset_name +"_M32x32_T30_InOut.h5"
            f = h5py.File(filename)
            data = f["data"][:, 1, :, :]

            IF = np.load(r'factors/'+ dataset_name +'.npy',allow_pickle=True)
            data = np.array(data)
            IF = IF.astype(np.float32)
            batch_size = 48
            channels = 1
            hidden_dim = [8,1]
            kernel_size = (3,3) # kernel size for two stacked hidden layer
            num_layers = 2  # number of stacked hidden layer
            # 数据维度：(seq, input_size)
            seq = data.shape[0]
            input_size = data.shape[1]
            # 定义每个样本中的时间步数量和预测步数
            seq_length = 12  # 过去的时间步数量
            pred_length = 1  # 预测的时间步数量
            Rs=rs
            height = Rs[0]
            width = Rs[1]
            x=[ele for ele in data]

            model = ConvGRU(input_size=(height, width),  # (h,w)
                            input_dim=channels,
                            hidden_dim=hidden_dim,
                            kernel_size=kernel_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bias=True,
                            dtype=dtype,
                            return_all_layers=False,
                            Rs=Rs
                            )
            model.to(device)
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            Us = [np.random.random([j, r]) for j, r in zip(list(x[0].shape), Rs)]
            best_loss = float('inf')

            for epoch in range(30):  # 进行模型训练
                # initilizer
                cores=get_cores(x, Us)
                train_data = []
                factor = []
                for i in range(seq-seq_length-pred_length+1):
                    train_data.append(cores[i:i + seq_length+1])
                for i in range(seq_length+pred_length-1,seq):
                    factor.append(IF[i])
                train_data=np.array(train_data)
                train_data = torch.tensor(train_data, dtype=torch.float32)
                factor=np.array(factor)
                factor = torch.tensor(factor, dtype=torch.float32)
                dataset = torch.utils.data.TensorDataset(train_data[:, :, :,:],factor[:, :])
                dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False,drop_last=True)

                # get core_pred
                k=20
                if epoch == 0:
                    k= 10
                for i in range(k):
                    loss_list=[]
                    cores_pred_list=[]
                    for ele in dataloader:
                        inputs=ele[0]
                        factor=ele[1]
                        inputs=inputs[:,:-1,:,:].reshape(batch_size,seq_length,1,Rs[0],Rs[1])
                        labels=inputs[:,-1,:,:].reshape(batch_size,pred_length,1,Rs[0],Rs[1])

                        _, cores_pred = model(inputs.to(device),factor.to(device),Rs ,Factors=False)
                        cores_pred_list.append(cores_pred)
                        loss = loss_function(cores_pred, labels.to(device))
                        loss_list.append(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    print(f"dataset_name: {dataset_name}, Rs: {Rs}, epoch: {epoch}, k: {i}, loss: {np.array(loss_list).mean():.4}  \n")
                    if epoch>=8 and np.array(loss_list).mean()< best_loss:
                        best_loss=np.array(loss_list).mean()
                        print(f"best_loss={best_loss}")
                        torch.save(model.state_dict(), f"save/factors/{dataset_name}/{Rs}-best_model2.pth")
                        np.save(f"save/factors/{dataset_name}/{Rs}-US[0]2.npy", Us[0])
                        np.save(f"save/factors/{dataset_name}/{Rs}-US[1]2.npy", Us[1])


                cores_pred_list=[ele.cpu().detach().numpy() for ele in cores_pred_list]
                cores_pred_list=np.array(cores_pred_list).reshape(-1,Rs[0],Rs[1])
                cores_pred_list = [ele for ele in cores_pred_list]

                for n in range(len(Rs)):
                    unfold_cores=update_cores( n, Us, x, cores, cores_pred_list,seq_length)
                    cores = get_fold_tensor(unfold_cores, n, cores_pred_list[0].shape)
                    Us=update_Us(Us, x, unfold_cores, n,seq_length)

            model.load_state_dict(torch.load(f"save/factors/{dataset_name}/{Rs}-best_model2.pth"))
            Us[0] = np.load(f"save/factors/{dataset_name}/{Rs}-US[0]2.npy", allow_pickle=True)
            Us[1] = np.load(f"save/factors/{dataset_name}/{Rs}-US[1]2.npy", allow_pickle=True)

            cores = get_cores(x, Us)
            train_data = []
            factor=[]
            for i in range(seq - seq_length - pred_length + 1):
                train_data.append(cores[i:i + seq_length + 1])
            for i in range(seq_length + pred_length - 1, seq):
                factor.append(IF[i])
            train_data = np.array(train_data)
            train_data = torch.tensor(train_data, dtype=torch.float32)
            factor = np.array(factor)
            factor = torch.tensor(factor, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(train_data[-48*7:, :, :, :], factor[-48*7:,:])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, drop_last=True)

            model.eval()
            pred = []
            x=[]
            mae=[]
            rmse=[]
            with torch.no_grad():
                for ele in dataloader:
                    inputs = ele[0][:, :-1, :, :]
                    labels = ele[0][:, -1, :, :]
                    factor = ele[1]
                    inputs = inputs.reshape(batch_size,seq_length, 1, Rs[0],Rs[1])
                    _, cores_pred = model(inputs.to(device),factor.to(device),Rs ,Factors=False)
                    cores_pred = list(cores_pred.cpu().reshape(-1,Rs[0],Rs[1]))
                    for ele2,ele3 in zip(cores_pred,labels):
                        a=tl.tenalg.multi_mode_dot(ele2.numpy(), Us)
                        b=tl.tenalg.multi_mode_dot(ele3.numpy(), Us)
                        pred.append(a)
                        x.append(b)
                        mae.append(MAE(a,b))
                        rmse.append(RMSE(a, b))
            print(f" Rs={Rs} MAE={MAE(pred, x)}\n RMSE={RMSE(pred, x)} \n ")
            with open('save/record—factors2.txt', 'a') as file:
                t2 = time.time()
                T2 = time.time()
                file.write(f"{dataset_name}  Rs={rs} \n")
                file.write(f"MAE={MAE(pred,x)}\n RMSE={RMSE(pred, x)} \n '程序运行时间:{(t2 - t)/60}分钟'\n")

