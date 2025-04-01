import torch
import os
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score,f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.font_manager import FontProperties
from sklearn.utils import shuffle
def read_file(path):
    """
    读取文本文件进行预处理
    :param path: 文件路径
    :return: 分词后的数组
    """
    if 'training_label' in path: #有标签的训练数据
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
            print(lines[:5])
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]
        return x, y
    elif 'training_nolabel' in path:#没有标签的训练数据
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x
    else:                           #测试数据
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            x = ["".join(line[1:].strip('\n').split(",")) for line in lines[1:]]
            x = [item.split(' ') for item in x]
        return x

def genfig(epochs,loss,precision,accuracy,recall,f1,path,mode):
    # 创建一个图形窗口，并指定figsize
    fig, ax1 = plt.subplots(figsize=(16, 9))
    # 绘制精确率、准确率和召回率曲线
    ax1.plot(epochs, precision, label='Precision', marker='o', color='b')
    ax1.plot(epochs, accuracy, label='Accuracy', marker='x', color='g')
    ax1.plot(epochs, recall, label='Recall', marker='s', color='r')
    ax1.plot(epochs, f1, label='F1_score', marker='d', color='orange')
    # 设置标题和轴标签
    ax1.set_title(f'{mode}:P,Acc,R,F1 and Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Score')
    # 添加图例
    ax1.legend(loc='upper left')
    # 创建第二个y轴并绘制损失曲线
    ax2 = ax1.twinx()
    ax2.plot(epochs, loss, label='Loss', marker='d', color='purple')
    ax2.set_ylabel('Loss')
    # 添加损失曲线的图例
    ax2.legend(loc='upper right')
    # 保存图形，并指定dpi
    fig.savefig(f'{path}/{mode}.png', dpi=100)
    # 显示图形
    plt.show()

def gen_cm(outputs,labels,path):
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()
    font_path = 'C:\Windows\Fonts\msyh.ttc'
    font_prop = FontProperties(fname=font_path)
    cm = confusion_matrix(labels, outputs)
    tn, fp, fn, tp = cm.ravel()
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
    fig, ax = plt.subplots(figsize=(8, 4))
    # 绘制表格
    table_data = [
        ["", "Vul", "Non_Vul"],
        ["Pre_Vul", f"TP: {tp}", f"FP: {fp}"],
        ["Pre_non_Vul", f"FN: {fn}", f"TN: {tn}"]
    ]        
    # 创建表格
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    # 设置颜色
    colors = [['w', '#f4cccc', '#f4cccc'], ['#d9ead3', 'w', 'w'], ['#d9ead3', 'w', 'w']]
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i][j])
            cell.set_text_props(fontproperties=font_prop)
    # 隐藏坐标轴
    ax.axis('off')
    # 添加标题
    plt.title('Confusion Matrix')
    fig.savefig(f'{path}/cm.png', dpi=100)
    plt.show()
    
def evaluate(outputs, labels):
    """
    分析结果
    :param outputs: 模型的输出
    :param labels: 数据集的标签
    :return: 正确的数目
    """
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    # 计算精确率
    precision = precision_score(labels, outputs,zero_division=1)
    # 计算准确率
    accuracy = accuracy_score(labels, outputs)
    # 计算召回率
    recall = recall_score(labels, outputs,zero_division=1)
    # 计算F1
    f1score = f1_score(labels,outputs,zero_division=1)
    #print(f"Recall: {recall:.4f}")
    #corrects = torch.sum(torch.eq(outputs, labels)).item()

    return precision,accuracy,recall,f1score

#读取智能合约部分
def read_contract(path,style):
    count = 0
    train_text = []
    label = []
    sourcecode_directory = path+'/'+style
    data_y_path = os.path.join(path, f'{style}.txt')
    #读取文件夹合约名称
    
    
    file = os.listdir(sourcecode_directory)
    file.sort(key=lambda x:int(x[:-4]))
    
    #dataset2
    labels = pd.read_csv(data_y_path)
    #按照 Filename 的数字部分进行排序,对齐标签和数据
    labels['Prefix'] = labels['Filename'].apply(lambda x: int(x.split('.')[0]))
    #根据 Prefix 列进行升序排序
    sorted_labels_df = labels.sort_values(by='Prefix').drop(columns=['Prefix'])
    labels_ = sorted_labels_df['Ground Truth'].tolist()
    
    
    #dataset3
    # labels = pd.read_csv(data_y_path,header=None)
    # labels_ = labels.values.flatten()
    
   

    for filename in file:
        if filename.endswith('.txt'):  # 确保文件是.txt格式的
            count += 1
            # 构建.txt文件的完整路径
            file_path = path + f'/{style}/' + f'{filename}'
            # 读取.sol文件内容
            with open(file_path, 'r') as f:
                source_code = f.read()
            # 使用Word2Vec将文本转换为向量表示
            #words = source_code.split()  # 将文本拆分为单词
            train_text.append(source_code)
    # with open(data_y_path, 'r') as f:
    #         labels = f.read().split()
    return train_text,labels_#,file

#增大语料库:
def read_morecontract(path):
    count = 0
    train_text = []
    for subdir in ['undependency','dependency']:
        with open('word2vec_trianlog.txt','w') as errorlog:
            for tp in os.listdir(path):
                contract_path = os.path.join(path,tp,subdir)
                for filename in os.listdir(contract_path):
                    if filename.endswith('.txt'):  # 确保文件是.txt格式的
                        count += 1
                        # 构建.sol文件的完整路径
                        file_path = os.path.join(contract_path, filename)
                        # 读取.sol文件内容
                        try:
                            with open(file_path, 'r',encoding='utf-8') as f:
                                source_code = f.read()
                            # 使用Word2Vec将文本转换为向量表示
                        except Exception as e:
                            error_message = f'error:{contract_path}{filename}: {e}\n'
                            print(error_message)  
                            errorlog.write(error_message)                  
                        words = source_code.split()  # 将文本拆分为单词
                        train_text.append(words)
    return train_text


def load_texts_from_directories(base_folder,sliced):

    texts = []
    labels = []
    
    for label, subdir in enumerate(['undependency', 'dependency']):
        folder_path = os.path.join(base_folder, subdir)
        for filename in os.listdir(folder_path):
            if filename.endswith(sliced):
                try:
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read().strip()
                        texts.append(text)
                        labels.append(label)  # 0 for undependency, 1 for dependency
                except UnicodeDecodeError:
                    continue

    # 打乱顺序
    texts, labels = shuffle(texts, labels, random_state=42)
    return texts, labels

import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, lstm_output):
        device = lstm_output.device
        self.scale = self.scale.to(device)
        # lstm_output: (batch_size, seq_len, hidden_dim * 2) 因为是双向的，所以维度是hidden_dim * 2
        query = lstm_output  # (batch_size, seq_len, hidden_dim * 2)
        key = lstm_output  # (batch_size, seq_len, hidden_dim * 2)
        value = lstm_output  # (batch_size, seq_len, hidden_dim * 2)
        
        # 计算点积注意力
        scores = torch.bmm(query, key.transpose(1, 2)) / self.scale  # (batch_size, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 加权和
        weighted_sum = torch.bmm(attention_weights, value)  # (batch_size, seq_len, hidden_dim * 2)
        return weighted_sum, attention_weights
