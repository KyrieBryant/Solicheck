import time
import psutil
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import os
from sklearn.model_selection import train_test_split
from model import BiLSTMModel_ScaledAttention,CombinedModel_add,CombinedModel_atadd,CodeBERTModel,CombinedModel_fusion,EarlyStopping
from data_loader import SoliCheckDataset
from utils import read_file, evaluate,read_contract,genfig,gen_cm,load_texts_from_directories
from pre_processing import DataPreprocess,DataPreprocess_Fasttext, DataPreprocess_bert
import matplotlib.pyplot as plt
import random
import numpy as np
import datetime
from collections import Counter
from imblearn.over_sampling import SMOTE
#                                       solicheck-w2v/ft
# 获取当前时间戳并格式化为"年-月-日"字符串
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# TYPE:reentrancy,origin,loop            
TYPE = 'reentrancy' 
is_data_aug = False
#是否联合模型
is_combine = True
#smote
is_smote = True 
#attention
attention = True
add_attention = False
scaled_attention = True
#加权系数
alpha = 0.5 
# FastText,W2v
# dataset_txt,dataset_func
data = 'dataset_func'
base_folder = f'./{data}/{TYPE}'
model_dir = './model'
sliced = '.txt'

#word2vec模型文件路径\
#w2v,ft
tool = 'w2v'


w2v_path = './model/newdataset/w2v_new_dataset_768.model'
fast_path = './model/newdataset/fasttext_model_768.model'
pt_file = ''
# 定义句子长度、是否固定 embedding、batch 大小、定义训练次数 epoch、learning rate 的值、model 的保存路径
requires_grad = False
sen_len = 300
batch_size = 32
epochs = 100
lr = 0.001
#LSTM
embedding_dim=768
input_dim = 768 
hidden_dim=384#因为是BiLSTM所以减半
num_layers=2
dropout=0.5
result_folder = ''



w2v = 1


def main():
    seed_everything(777)


                            # data pre_processing
    data_x, data_y = load_texts_from_directories(base_folder,sliced)
    start = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss  # 返回值单位是字节

    if tool == 'w2v':
        result_folder = f'./results/{TYPE}_w2v/{timestamp}{data}'
        preprocess = DataPreprocess(data_x, sen_len, w2v_path)
        print('w2v model loaded')
    elif tool == 'ft':
        result_folder = f'./results/{TYPE}_ft/{timestamp}{data}'
        preprocess = DataPreprocess_Fasttext(data_x, sen_len, fast_path)
        print('fasttext model loaded')
    mem_after = process.memory_info().rss
    mem_diff = mem_after - mem_before  # 增量内存    
    end = time.time()
    print(f"数据加载平均时间:{(end-start)/len(data_y):.6f}s")
    print(f"内存消耗:{(mem_diff)/1024/1024:.2f}MB")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    embedding = preprocess.make_embedding()#返回的是word2vec中词汇表的嵌入向量的矩阵
    data_x = preprocess.sentence_word2idx()#返回data_x经过查字典得到的索引
    data_y = preprocess.labels2tensor(data_y)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=5)
    print(f"split data : train:0.8 , test:0.2 ")
    

    
    if is_combine:
        lstm = BiLSTMModel_ScaledAttention(
        embedding,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        requires_grad=requires_grad
    ).to(device)
        codebert = CodeBERTModel()
        model = CombinedModel_atadd(lstm,codebert,hidden_dim=hidden_dim).to(device)
    else:
        model = BiLSTMModel_ScaledAttention(
        embedding,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        requires_grad=requires_grad
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
        shape_ = shape
        print(f"No Smote : {shape}")
    
    # 构造Dataset,训练集和验证集
    train_dataset = SoliCheckDataset(x_train, y_train)
    val_dataset = SoliCheckDataset(x_test, y_test)

    # preparing the training loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Training loader prepared.',f'the length is {len(train_dataset)}')
    # preparing the validation loader
    val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
    print('Validation loader prepared.',f'the length is {len(val_dataset)}')
    
 
    # 返回model中的参数的总数目
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    # loss function is binary cross entropy loss, 常见的二分类损失函数
    criterion = nn.BCELoss()
    #可以试试BCEWithLogitsLoss
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # 每10个epoch降低学习率

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
    
    early_stopping = EarlyStopping(
    patience=10, 
    delta=0.0, 
    verbose=True, 
    #path="./model/2024.12.28/{}_{}_best_model.pt".format(TYPE, tool)
    )
    # 初始化最佳 F1 分数和最佳 epoch
    best_epoch = -1
    time_cost = []
    gpu_cost = []
                                    # run epochs
    for epoch in range(epochs):
        xis.append(epoch+1)
                            # train for one epoch
        train_start = time.time()
        gpu_mem_before = torch.cuda.memory_allocated(device)#监控GPU内存                    
        temp_loss,p_temp,acc_temp,r_temp,f_temp = train(train_loader, model, criterion, optimizer, epoch,attention)
        train_end = time.time()
        gpu_mem_after = torch.cuda.memory_allocated(device)
        time_cost.append(train_end-train_start)
        gpu_cost.append(gpu_mem_after-gpu_mem_before)
        print(f"训练时间:{train_end-train_start:.2f}   GPU占用:{(gpu_mem_after-gpu_mem_before)/1024/1024:.2f}MB")
        
        loss.append(temp_loss)
        train_p.append(p_temp)
        train_acc.append(acc_temp)
        train_r.append(r_temp)
        train_f.append(f_temp)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
                        # evaluate on validation set
        total_p,total_acc,total_r,total_f,outputs,labels = validate(val_loader, model, criterion)
        Val_p.append(total_p)
        Val_acc.append(total_acc)
        Val_r.append(total_r)
        Val_f.append(total_f)
        print(f'        Learning Rate: {current_lr}')
        
        early_stopping(total_f, model)
        if early_stopping.early_stop:
            print("早停机制触发，停止训练。")
            break
        
        if total_f > best_f:
            best_f = total_f
            best_epoch = epoch+1
            torch.save(model, "{}/{}_{}_ckpt.model".format(model_dir,TYPE,tool))
            print('saving model with val_av_f {:.4f}'.format(total_f))
            
        if epoch == epochs-1:
            gen_cm(outputs,labels,result_folder)
    print(f'vulnerability:{TYPE}')    
    print(f'The final best val_av_f is :{best_f},epoch:{best_epoch}')
    print(f'time cost :{sum(time_cost)/len(time_cost):.2f}s  GPU cost :{sum(gpu_cost)/1024/1024:.2f}MB')
    #model.load_state_dict(torch.load("{}/{}_{}_best_model.pt".format(model_dir, TYPE, tool)))

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
        LSTM_alpha:{alpha}
        
        vectorlizer:{tool}
        
        sol or txt:{sliced}
        
        dataload_time_cost:{end-start}
        train_time_cost:{sum(time_cost)/len(time_cost):.2f}s  
        GPU cost :{sum(gpu_cost)/1024/1024:.2f}MB
        ''')
    with open(f'{result_folder}/info.txt','w') as info:
        info.write('loss\ttra_p\ttra_a\ttra_r\ttra_f\tval_p\tval_acc\tval_r\tval_f\n')
        for l,tp,ta,tr,tf,vp,va,vr,vf in zip(loss,100*train_p,100*train_acc,100*train_r,100*train_f,100*Val_p,100*Val_acc,100*Val_r,100*Val_f) :
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


def train(train_loader, model, criterion, optimizer, epoch,attention):
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
    alpha=0
    for i, (inputs, labels) in enumerate(train_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)  # 类型为float
        # 2. 清空梯度
        optimizer.zero_grad()
        outputs,alpha= model(inputs)
        outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()    
        p,a,r,f = evaluate(outputs.clone().detach(), labels)
        loss.backward()
        
        #统计正确个数计算正确率
        #correct= evaluate(outputs.clone().detach(), labels)
        # total_acc += (correct / batch_size)
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
        #if i % 20 == 0:
    print('Train | Epoch{}:  Loss:{:.5f}  alpha:{:.4f}  train_av_p: {:.2f}%   train_av_acc:   {:.2f}% train_av_r:   {:.2f}% train_av_f: {:.2f}%'.format(epoch + 1,total_loss / train_len, alpha.data,total_p*100 / train_len,total_acc*100 / train_len,total_r*100 / train_len,total_f*100 / train_len))
    return total_loss/ train_len,p_temp,acc_temp,r_temp,f_temp

def validate(val_loader, model, criterion):
    model.eval()  # 將 model 的模式设定为 eval，固定model的参数

    val_len = len(val_loader)

    with torch.no_grad():
        total_loss,total_p, total_acc,total_r = 0, 0,0,0
        for i, (inputs, labels) in enumerate(val_loader):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            outputs,_= model(inputs)
            outputs = outputs.squeeze()
            # 3. 计算损失

            # 4. 预测结果
            p,a,r,f = evaluate(outputs, labels)

        print("Valid |                                        Val_av_p: {:.2f}%    Val_av_acc:    {:.2f}%  Val_av_r:    {:.2f}%  Val_av_f:  {:.2f}% ".format(p*100,a*100,r*100,f*100))
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        
    return p,a,r,f,outputs,labels
 

if __name__ == '__main__':
    main()


