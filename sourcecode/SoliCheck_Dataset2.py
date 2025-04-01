import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import os
import time
import psutil
from sklearn.model_selection import train_test_split
from model import CombinedModel_1,CodeBERTModel,LSTM_bert,CombinedModel_12,EarlyStopping,BiLSTM_bert
from data_loader import SoliCheckDataset
from utils import read_file, evaluate,read_contract,genfig,gen_cm,load_texts_from_directories
from pre_processing import DataPreprocess,DataPreprocess_bert,DataPreprocess_Fasttext
import matplotlib.pyplot as plt
import random
import numpy as np
import datetime
from collections import Counter
from imblearn.over_sampling import SMOTE
#                                   本文是BiLSTM+Codebert
# 获取当前时间戳并格式化为"年-月-日"字符串
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



#                                           嵌入层改为bert
# BN,IO,RE,EF,UC,SE,DE,TD,RE_TEST                  
TYPE = 'SE'
is_data_aug = False
#是否联合模型
is_combine = True
#smote
is_smote = True 
#attention
add_attention = False
scaled_attention = True
attention = True
#加权系数
result_folder = f'./results2/{TYPE}/{timestamp}'
os.makedirs(result_folder)
#模型文件路径
#bert,w2v,ft
embed = 'bert'
w2v_path = './model/contract_word2vec.model'
fast_path = './model/newdataset/fasttext_model_768.model'

# 定义句子长度、是否固定 embedding、batch 大小、定义训练次数 epoch、learning rate 的值、model 的保存路径
requires_grad = False
sen_len = 500
batch_size = 32
epochs = 100
lr = 0.001
#LSTM
embedding_dim=768
input_dim = 768 
hidden_dim=384#因为是BiLSTM所以减半
num_layers=2
dropout=0.5
sliced = '.txt'
model_dir = './model_Dataset2'

# model_dir = os.path.join(path_prefix, 'model/')

base_folder = f'./Dataset2_mini'

def main():
    seed_everything(777)
    print(f"{TYPE}:使用bert准备数据")
    #                                   基本参数

    #                                   准备数据
    # 调用函数并读取数据
    #不行，专门给bert写的
                                # data pre_processing
    data_x, data_y = read_contract(base_folder,TYPE)

    start = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    preprocess = DataPreprocess_bert(data_x, sen_len)
    
    # 调用函数并将数据转化为tensor
    embedding = preprocess.make_embedding()#返回的是bert中词汇表的嵌入向量的矩阵
    data_x = preprocess.sentence_word2idx()#返回data_x经过查字典得到的索引
    data_y = preprocess.labels2tensor(data_y)
    mem_after = process.memory_info().rss
    mem_diff = mem_after - mem_before  # 增量内存 
    end = time.time()
    print(f"数据加载平均时间:{(end-start)/len(data_y):.2f}s")
    print(f"内存消耗:{(mem_diff)/1024/1024:.2f}MB")
    # split data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=5)
    print(f"split data : train:0.8 , test:0.2 ")
    

     
    if is_combine:
        model = LSTM_bert(
        './bert-uncased',
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
        codebert = CodeBERTModel()
        model = CombinedModel_12(model,codebert,hidden_dim=hidden_dim).to(device)
    else:
        model = BiLSTM_bert(
        './bert-uncased',
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
        
    if is_smote == True:
        #SMOTE 平衡数据集部分
        print(f"————————————————————SMOTE————————————————————————") 
        data_x_np = x_train.numpy()
        data_y_np = y_train.numpy()
        shape = Counter(data_y_np)
        print(f"Original dataset shape: {shape}") 
        # 创建SMOTE对象并设置参数
        
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)

        # 进行过采样
        x_train, y_train = smote.fit_resample(data_x_np, data_y_np)
        shape_ = Counter(y_train)
        print(f"After smote dataset shape: {shape_}") 
    else :
        data_y_np = y_train.numpy()
        shape = Counter(data_y_np)
        print(f"No Smote : {shape}")
    
    # 构造Dataset,训练集和验证集
    train_dataset = SoliCheckDataset(x_train, y_train)
    val_dataset = SoliCheckDataset(x_test, y_test)

    # preparing the training loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Training loader prepared.',f'the length is {len(train_dataset)}')
    # preparing the validation loader
    val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
    print('Validation loader prepared.',f'the length is {len(val_dataset)},count:{Counter(y_test.numpy())}')
    
 
    # 返回model中的参数的总数目
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    # loss function is binary cross entropy loss, 常见的二分类损失函数
    criterion = nn.BCELoss()
    #可以试试BCEWithLogitsLoss
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # 每10个epoch降低学习率

    best_f = 0.
    loss = []
     
    Val_p = []
    Val_acc = []
    Val_r = []
    Val_f = []
    
    train_p = []
    train_acc = []
    train_r = []
    train_f = []
    xis = []
    #时间和内存
    time_cost = []
    gpu_cost = []
    best_epoch = 0
    early_stopping = EarlyStopping(
    patience=10, 
    delta=0.0, 
    verbose=True, 
    #path="./model/2024.12.28/{}_{}_best_model.pt".format(TYPE, 'bert')
    )
                                # run epochs
    for epoch in range(epochs):
        epoch_start_time = time.time() #监控时间
        gpu_mem_before = torch.cuda.memory_allocated(device)#监控GPU内存
        xis.append(epoch+1)
                            # train for one epoch
        temp_loss,p_temp,acc_temp,r_temp,f_temp = train(train_loader, model, criterion, optimizer, epoch,attention,preprocess)

        loss.append(temp_loss)
        train_p.append(p_temp)
        train_acc.append(acc_temp)
        train_r.append(r_temp)
        train_f.append(f_temp)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0] 
        epoch_end_time = time.time()
        gpu_mem_after = torch.cuda.memory_allocated(device)
        gpu_mem_used = gpu_mem_after - gpu_mem_before
        epoch_duration = epoch_end_time - epoch_start_time
        
        time_cost.append(epoch_duration)
        gpu_cost.append(gpu_mem_used/1024/1024)
        
        print(f"\tTime: {epoch_duration:.2f} s   - GPU Memory Usage: +{gpu_mem_used/1024/1024:.2f} MB\n")
        
                        # evaluate on validation set
        total_p,total_acc,total_r,total_f,outputs,labels = validate(val_loader, model, criterion,preprocess)
        Val_p.append(total_p)
        Val_acc.append(total_acc)
        Val_r.append(total_r)
        Val_f.append(total_f)
        early_stopping(total_f, model)
        print(f'\tLearning Rate:{current_lr}')
        if early_stopping.early_stop:
            print("******early stopping******")
            break
        if total_f > best_f:
            # 如果 validation 的结果好于之前所有的结果，就把当下的模型保存
            best_epoch = epoch + 1
            best_f = total_f
            torch.save(model, "{}/{}_ckpt.model".format(model_dir,TYPE))
            print('saving model with val_av_f {:.4f}'.format(total_f))
            
        #最后一个epoch保存混淆矩阵
        if epoch == epochs-1:
            gen_cm(outputs,labels,result_folder)    
    print(f'The final best val_av_f is :{best_f:.4f}, at epoch {best_epoch}')    
    print(f'average time_cost:{sum(time_cost)/len(time_cost):.2f}s, average gpu_cost:{sum(gpu_cost):.2f}MB')
    #保存实验数据
    
    
    with open(f'{result_folder}/paras.txt','w') as para:
        para.write(f'''
    data:
        contract_TYPE:{TYPE}
        requires_grad={requires_grad}
        sen_len={sen_len}
        batch_size={batch_size}
        epochs={epochs}
        lr={lr}
        LSTM:
        embedding_dim={embedding_dim}
        hidden_dim={hidden_dim}
        num_layers={num_layers}
        dropout={dropout}
        
        result:
        best_val_f={best_f}
        
        smote:{is_smote}
        shape before smote :{shape}
        shape after smote  :{shape_}
        
        attention:{attention}
        add_attention:{add_attention}
        sacled_attention:{scaled_attention}
        
        data_Augmentation:{is_data_aug}
        
        combine_model:{is_combine}
        slice:{sliced}
        
        dataload_time/sol_cost:{(end-start/len(data_y)):.2f}
        time/epoch_cost:{sum(time_cost)/len(time_cost):.2f}s
        gpu_cost:{sum(gpu_cost):.2f}MB
        ''')
    with open(f'{result_folder}/info.txt','w') as info:
        info.write('loss\ttra_p\ttra_a\ttra_r\ttra_f\tval_p\tval_acc\tval_r\tval_f\n')
        for l,tp,ta,tr,tf,vp,va,vr,vf in zip(loss,train_p,train_acc,train_r,train_f,Val_p,Val_acc,Val_r,Val_f) :
            info.write(f'{l:.4f}\t{tp:.4f}\t{ta:.4f}\t{tr:.4f}\t{tf:.4f}\t{vp:.4f}\t{va:.4f}\t{vr:.4f}\t{vf:.4f}\t\n')
            
    #作图展示loss和P,Acc,R的变化
    #训练数据
    mode = 'train'
    genfig(xis,loss,train_p,train_acc,train_r,train_f,result_folder,mode)
    #验证数据
    mode = 'val'
    genfig(xis,loss,Val_p,Val_acc,Val_r,Val_f,result_folder,mode)
    print(f'info&png have saved')
    

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(train_loader, model, criterion, optimizer, epoch,attention,preprocess):
    # 將 model 的模式设定为 train，这样 optimizer 就可以更新 model 的参数
    model.train()
    p_temp = 0
    acc_temp = 0
    r_temp = 0
    f_temp = 0
    
    train_len = len(train_loader)
    total_loss =0
    
    total_acc = 0
    total_p = 0
    total_r = 0
    total_f = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)  # 类型为float
        # 2. 清空梯度
        optimizer.zero_grad()        
        attention_mask = (inputs != preprocess.tokenizer.pad_token_id).long().to(device)

        outputs,alpha= model(inputs,attention_mask)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()    
        p,a,r,f = evaluate(outputs.clone().detach(), labels)
        loss.backward()
        total_p += p
        total_acc += a
        total_r += r
        total_f += f
        # if correct * 100 / batch_size > acc_temp:
        #     acc_temp = correct * 100 / batch_size
        if p > p_temp:
            p_temp = p
        if a > acc_temp:
            acc_temp = a
        if r > r_temp:
            r_temp = r
        if f > f_temp:
            f_temp = f
        # 6. 反向传播
        #loss.backward()
        # 7. 更新梯度
        optimizer.step()

    print('Train | Epoch{}:  Loss:{:.5f}  alpha:{:.4f}  train_av_p: {:.4f}   train_av_acc:   {:.4f} train_av_r:   {:.4f} train_av_f:   {:.4f}'.format(epoch + 1,total_loss / train_len, alpha.data ,total_p / train_len,total_acc / train_len,total_r / train_len,total_f/train_len))
    return total_loss/ train_len,p_temp,acc_temp,r_temp,f_temp

def validate(val_loader, model, criterion,preprocess):
    model.eval()  # 將 model 的模式设定为 eval，固定model的参数

    val_len = len(val_loader)

    with torch.no_grad():
        total_loss,total_p, total_acc,total_r = 0, 0,0,0
        for i, (inputs, labels) in enumerate(val_loader):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            attention_mask = (inputs != preprocess.tokenizer.pad_token_id).long().to(device)
            outputs,_= model(inputs,attention_mask)
            outputs = outputs.squeeze()
            # 3. 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # 4. 预测结果
            p,a,r,f = evaluate(outputs, labels)
            #correct= evaluate(outputs, labels)
            #total_acc += (correct / batch_size)
        print("Valid |           Loss:{:.5f}    Val_av_p: {:.4f}    Val_av_acc: {:.4f}  Val_av_r: {:.4f}  Val_av_f: {:.4f} ".format(total_loss / val_len, p,a,r,f))
    print('--------------------------------------------------------------------------------------\n')

    return p,a,r,f,outputs,labels


if __name__ == '__main__':
    main()


