import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification,AutoModel,AutoTokenizer


class CodeBERTModel(torch.nn.Module):
    def __init__(self, model_name="./codebert-base", num_labels=1):
        super(CodeBERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs.last_hidden_state[:,-1,:]  



class LSTMModel(torch.nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=False):
        super(LSTMModel, self).__init__()
        # 这是一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
        # embedding: (嵌入字典的大小, 每个嵌入向量的维度)embedding是word2vec字典的嵌入向量矩阵
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        # 将embedding不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，并将这个参数绑定到module里面，成为module中可训练的参数。
        self.embedding.weight = torch.nn.Parameter(embedding, requires_grad=requires_grad)
        self.LSTM = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.LSTM(inputs)
        #取所有时间步的加权和,感觉不行，增加了FP和TN的数量
        #x = x.mean(dim=1)
        #取最后一个时间步的结果
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
    
        # x.shape = (batch_size, seq_len, hidden_size)
        # 取用 LSTM 最后一个的 hidden state
        
        
        # #只用线性层试试
        # inputs = self.embedding(inputs)
        # #inputs.size = (batch_size,seq_len,embedding)
        # inputs = inputs[:, -1, :]
        # x = self.classifier_line(inputs)
   
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        #self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(0)

        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)

        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob
               
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        energy = self.attn(lstm_output)  # (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(energy)      # Apply tanh activation
        attention_scores = self.context(energy).squeeze(-1)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # 加权平均 LSTM 输出
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # (batch_size, 1, hidden_dim)
        weighted_sum = weighted_sum.squeeze(1)  # (batch_size, hidden_dim)
        
        return weighted_sum, attention_weights

class LSTMModel_AddAttention(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=False):
        super(LSTMModel_AddAttention, self).__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # 获取词嵌入
        inputs = self.embedding(inputs)
        # 获取LSTM输出
        lstm_output, _ = self.LSTM(inputs)
        # 获取加权的LSTM输出
        attention_output, attention_weights = self.attention(lstm_output)
        # 通过分类器
        x = self.classifier(attention_output[:, -1, :])
        return x
    
class LSTMModel_ScaledAttention(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=False):
        super(LSTMModel_ScaledAttention, self).__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = ScaledDotProductAttention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # 获取词嵌入
        inputs = self.embedding(inputs)
        # 获取LSTM输出
        lstm_output, _ = self.LSTM(inputs)
        # 获取加权的LSTM输出
        attention_output, attention_weights = self.attention(lstm_output)
        # 使用最后的时间步（或平均加权输出）
        x = self.classifier(attention_output[:, -1, :])
        return x

class BiLSTM_ScaledAttention(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=False):
        super(BiLSTM_ScaledAttention, self).__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        #self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)
        self.BiLSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attention = ScaledDotProductAttention(hidden_dim * 2)  # 因为是双向的，所以hidden_dim需要乘以2
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 由于是双向LSTM，所以hidden_dim需要乘2
        self.activation = nn.Sigmoid()
    def forward(self, inputs):
        # 获取词嵌入
        inputs = self.embedding(inputs)
        # 获取BiLSTM输出
        #lstm_output, _ = self.BiLSTM(inputs)
        
        lstm_out, (hn, cn) = self.BiLSTM(inputs)
        
        # 获取加权的BiLSTM输出
        attention_output, attention_weights = self.attention(lstm_out)
        
        # 使用最后的时间步（或平均加权输出）
        x = attention_output[:, -1, :]
        return self.activation(self.fc(x))

#用于联合模型的lstm

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, lstm_output):
        device = lstm_output.device
        self.scale = self.scale.to(device)
        # lstm_output: (batch_size, seq_len, hidden_dim)
        query = lstm_output # (batch_size, seq_len, hidden_dim)
        key = lstm_output  # (batch_size, seq_len, hidden_dim)
        value = lstm_output  # (batch_size, seq_len, hidden_dim)
        
        # 计算点积注意力
        scores = torch.bmm(query, key.transpose(1, 2)) / self.scale  # (batch_size, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 加权和
        weighted_sum = torch.bmm(attention_weights, value)  # (batch_size, seq_len, hidden_dim)
        return weighted_sum, attention_weights

class LSTM_ScaledAttention(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=False):
        super(LSTM_ScaledAttention, self).__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = ScaledDotProductAttention(hidden_dim)

    def forward(self, inputs):
        # 获取词嵌入
        inputs = self.embedding(inputs)
        # 获取LSTM输出
        lstm_output, _ = self.LSTM(inputs)
        # 获取加权的LSTM输出
        attention_output, attention_weights = self.attention(lstm_output)
        # 使用最后的时间步（或平均加权输出）
        return attention_output[:, -1, :]

class CombinedModel_add(torch.nn.Module):
    def __init__(self, lstm_model, codebert_model, hidden_dim):
        super(CombinedModel_add, self).__init__()
        self.lstm_model = lstm_model
        self.codebert_model = codebert_model
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        # 分类器层，输入维度为hidden_dim，输出维度为1（用于二分类）
        
        self.classifier = torch.nn.Sequential(
            nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim*2, 1),  # 输入为hidden_dim，输出为1
            torch.nn.Sigmoid()  # 用Sigmoid进行二分类
        )
    
    def forward(self, input):
        lstm_output = self.lstm_model(input)  # LSTM模型输出，形状: (batch_size, hidden_dim)
        
        codebert_output = self.codebert_model(input)  # CodeBERT模型输出，形状: (batch_size, hidden_dim)
        # 可变加权求和（可以根据需要调整alpha的值）
        combined_output = self.alpha * lstm_output + (1 - self.alpha) * codebert_output
        # 输入分类器
        final_output = self.classifier(combined_output)  # 输出形状: (batch_size, 1)
        return final_output,self.alpha

class CombinedModel_atadd(torch.nn.Module):
    def __init__(self, lstm_model, codebert_model, hidden_dim):
        super(CombinedModel_atadd, self).__init__()
        self.lstm_model = lstm_model
        self.codebert_model = codebert_model
        # 分类器层，输入维度为hidden_dim，输出维度为1（用于二分类）
        self.attention = nn.Linear(hidden_dim * 4, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制两个模型的加权比率，可变权重
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()  # 如果使用 BCEWithLogitsLoss，移除此行
        )
    
    def forward(self, input):
        lstm_output = self.lstm_model(input)  # LSTM模型输出，形状: (batch_size, hidden_dim)
        codebert_output = self.codebert_model(input)  # CodeBERT模型输出，形状: (batch_size, hidden_dim)
        #拼接
        combined = torch.cat((lstm_output, codebert_output), dim=1)  # (batch_size, hidden_dim*4)
        weights = torch.sigmoid(self.attention(combined))
        # 可变加权求和（可以根据需要调整alpha的值）
        combined_output = weights * lstm_output + (1 - weights) * codebert_output
        # 输入分类器
        final_output = self.classifier(combined_output)  # 输出形状: (batch_size, 1)
        return final_output,self.alpha      #alpha拿来占位的，实际上并没有用到

class CombinedModel_fusion(torch.nn.Module):
    def __init__(self, lstm_model, codebert_model, hidden_dim):
        super(CombinedModel_fusion, self).__init__()
        self.lstm_model = lstm_model
        self.codebert_model = codebert_model
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制两个模型的加权比率，可变权重
        # 分类器层，输入维度为 hidden_dim * 4（LSTM 输出 + CodeBERT 输出），输出维度为1（用于二分类）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()  # 如果使用 BCEWithLogitsLoss，移除此行
        )
    
    def forward(self, input):
        lstm_output = self.lstm_model(input)  # LSTM模型输出，形状: (batch_size, hidden_dim)        
        codebert_output = self.codebert_model(input) # CodeBERT模型输出，形状: (batch_size, hidden_dim)
        # 拼接
        combined_features = torch.cat((lstm_output, codebert_output), dim=1)        
        # 输入分类器
        final_output = self.classifier(combined_features)  # 输出形状: (batch_size, 1)
        return final_output,self.alpha

class BiLSTMModel_ScaledAttention(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=False):
        super(BiLSTMModel_ScaledAttention, self).__init__()
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding, requires_grad=requires_grad)
        self.BiLSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attention = ScaledDotProductAttention(hidden_dim * 2)  # 因为是双向的，所以hidden_dim需要乘以2
    def forward(self, inputs):
        # 获取词嵌入
        inputs = self.embedding(inputs)
        # 获取BiLSTM输出
        #lstm_output, _ = self.BiLSTM(inputs)
        
        lstm_out, (hn, cn) = self.BiLSTM(inputs)
        
        # 获取加权的BiLSTM输出
        attention_output, attention_weights = self.attention(lstm_out)
        
        # 使用最后的时间步（或平均加权输出）
        x = attention_output[:, -1, :]
        return x
    
from transformers import BertModel, BertTokenizer

class LSTM_bert(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_layers, dropout=0.5):
        super(LSTM_bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=num_layers, bidirectional=True,batch_first=True, dropout=dropout)
        self.attention = ScaledDotProductAttention(hidden_dim*2)
        
    def forward(self, input_ids, attention_mask):
        # 使用BERT生成输入张量的嵌入
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取LSTM输出
        lstm_output, _ = self.LSTM(bert_output.last_hidden_state)
        
        # 获取加权的LSTM输出
        attention_output, attention_weights = self.attention(lstm_output)
        
        # 使用最后的时间步（或平均加权输出）
        return attention_output[:, -1, :]

class BiLSTM_bert(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_layers, dropout=0.5):
        super(BiLSTM_bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=num_layers, bidirectional=True,batch_first=True, dropout=dropout)
        self.attention = ScaledDotProductAttention(hidden_dim*2)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 占位
        self.classifier = torch.nn.Sequential(
            nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim*2, 1),  # 输入为hidden_dim，输出为1
            torch.nn.Sigmoid()  # 用Sigmoid进行二分类
        )
    def forward(self, input_ids, attention_mask):
        # 使用BERT生成输入张量的嵌入
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取LSTM输出
        lstm_output, _ = self.LSTM(bert_output.last_hidden_state)
        
        # 获取加权的LSTM输出
        attention_output, attention_weights = self.attention(lstm_output)
        return self.classifier(attention_output[:, -1, :]),self.alpha

class CombinedModel_1(torch.nn.Module):
    def __init__(self, lstm_model, codebert_model, hidden_dim):
        super(CombinedModel_1, self).__init__()
        self.lstm_model = lstm_model
        self.codebert_model = codebert_model
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制两个模型的加权比率，可变权重
        # 分类器层，输入维度为hidden_dim，输出维度为1（用于二分类）
        self.classifier = torch.nn.Sequential(
            nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim*2, 1),  # 输入为hidden_dim，输出为1
            torch.nn.Sigmoid()  # 用Sigmoid进行二分类
        )
    
    def forward(self, input,mask):
        lstm_output = self.lstm_model(input,mask)  # LSTM模型输出，形状: (batch_size, hidden_dim)
        codebert_output = self.codebert_model(input)  # CodeBERT模型输出，形状: (batch_size, hidden_dim)
        # 加权求和（可以根据需要调整alpha的值）
        combined_output = self.alpha * lstm_output + (1 - self.alpha) * codebert_output
        # 输入分类器
        final_output = self.classifier(combined_output)  # 输出形状: (batch_size, 1)
        return final_output,self.alpha


class CombinedModel_12(torch.nn.Module):
    def __init__(self, lstm_model, codebert_model, hidden_dim):
        super(CombinedModel_12, self).__init__()
        self.lstm_model = lstm_model
        self.codebert_model = codebert_model
        self.attention = nn.Linear(hidden_dim * 4, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制两个模型的加权比率，可变权重
        # 分类器层，输入维度为hidden_dim，输出维度为1（用于二分类）
        self.classifier = torch.nn.Sequential(
            nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim*2, 1),  # 输入为hidden_dim，输出为1
            torch.nn.Sigmoid()  # 用Sigmoid进行二分类
        )
    
    def forward(self, input,mask):
        lstm_output = self.lstm_model(input,mask)  # LSTM模型输出，形状: (batch_size, hidden_dim)
        codebert_output = self.codebert_model(input)  # CodeBERT模型输出，形状: (batch_size, hidden_dim)
        # 加权求和（可以根据需要调整alpha的值）
        weights = torch.sigmoid(self.attention(torch.cat((lstm_output, codebert_output), dim=1)))
        combined_output = weights * lstm_output + (1 - weights) * codebert_output
        # 输入分类器
        final_output = self.classifier(combined_output)  # 输出形状: (batch_size, 1)
        return final_output,self.alpha

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, verbose=False, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        score = score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, score)
            self.counter = 0

    def save_checkpoint(self, model, score):
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')

        #torch.save(model.state_dict(), self.path)
        self.best_model_state = model.state_dict()

