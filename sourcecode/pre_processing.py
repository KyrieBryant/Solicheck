import torch
from gensim.models import Word2Vec
import numpy as np
import fasttext.util
from gensim.models.fasttext import FastText
from transformers import BertTokenizer, BertModel

# 备注：
# 更改了  w2v_path

class DataPreprocess:
    def __init__(self, sentences, sen_len, w2v_path):
        self.sentences = sentences  # 句子列表
        self.sen_len = sen_len      # 句子的最大长度
        self.w2v_path = w2v_path    # word2vec模型路径
        self.index2word = []        # 实现index到word转换
        self.word2index = {}        # 实现word到index转换
        self.embedding_matrix = []

        # load word2vec.model
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def make_embedding(self):
        # 为model里面的单词构造word2index, index2word 和 embedding
        for i, word in enumerate(self.embedding.wv.vocab):
            #print('get word #{}'.format(i+1), end='\r')
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        self.embedding_matrix = np.array(self.embedding_matrix)
        self.embedding_matrix = torch.tensor(self.embedding_matrix)

        # 將"<PAD>"和"<UNK>"加进embedding里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))

        return self.embedding_matrix

    def add_embedding(self, word):
        # 将新词添加进embedding中
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def sentence_word2idx(self):
        sentence_list = []
        for i, sentence in enumerate(self.sentences):
            # 将句子中的单词表示成index
            sentence_index = []
            for word in sentence:
                if word in self.word2index.keys():
                    # 如果单词在字典中则直接读取index
                    sentence_index.append(self.word2index[word])
                else:
                    # 否则赋予<UNK>
                    sentence_index.append(self.word2index["<UNK>"])

            # 统一句子长度
            sentence_index = self.pad_sequence(sentence_index)
            #分片
            #sentence_index = self.slice_sequence(sentence_index,256)
            sentence_list.append(sentence_index)

        return torch.LongTensor(sentence_list)
   
            
    def pad_sequence(self, sentence):
        # 统一句子长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2index["<PAD>"])
        assert len(sentence) == self.sen_len

        return sentence

    def labels2tensor(self, y):
        y = [int(label) for label in y]

        return torch.LongTensor(y)


class DataPreprocess_Fasttext:
    def __init__(self, sentences, sen_len, ft_path):
        self.sentences = sentences  # 句子列表
        self.sen_len = sen_len      # 句子的最大长度
        self.ft_path = ft_path      # FastText模型路径
        self.index2word = []        # 实现index到word转换
        self.word2index = {}        # 实现word到index转换
        self.embedding_matrix = []

        # 加载FastText模型
        self.embedding = FastText.load(self.ft_path)
        self.embedding_dim = self.embedding.vector_size

    def make_embedding(self):
        # 为模型里面的单词构造word2index, index2word和embedding
        for i, word in enumerate(self.embedding.wv.vocab):
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        self.embedding_matrix = np.array(self.embedding_matrix)
        self.embedding_matrix = torch.tensor(self.embedding_matrix)

        # 将"<PAD>"和"<UNK>"加进embedding里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))

        return self.embedding_matrix

    def add_embedding(self, word):
        # 将新词添加进embedding中
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def sentence_word2idx(self):
        sentence_list = []
        for i, sentence in enumerate(self.sentences):
            # 将句子中的单词表示成index
            sentence_index = []
            for word in sentence:
                if word in self.word2index.keys():
                    # 如果单词在字典中则直接读取index
                    sentence_index.append(self.word2index[word])
                else:
                    # 否则赋予<UNK>
                    sentence_index.append(self.word2index["<UNK>"])

            # 统一句子长度
            sentence_index = self.pad_sequence(sentence_index)
            sentence_list.append(sentence_index)

        return torch.LongTensor(sentence_list)

    def pad_sequence(self, sentence):
        # 统一句子长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2index["<PAD>"])
        assert len(sentence) == self.sen_len

        return sentence

    def labels2tensor(self, y):
        y = [int(label) for label in y]

        return torch.LongTensor(y)
    
class DataPreprocess_bert:
    def __init__(self, sentences, sen_len, bert_model_name='bert-uncased'):
        self.sentences = sentences  # 句子列表
        self.sen_len = sen_len      # 句子的最大长度
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.embedding_dim = self.bert_model.config.hidden_size

    def make_embedding(self):
        # 这里不需要像Word2Vec那样手动构建embedding_matrix
        # BERT会动态生成每个输入句子的embedding
        return None

    def add_embedding(self, word):
        # BERT不需要手动添加单词
        pass

    def sentence_word2idx(self):
        sentence_list = []
        for sentence in self.sentences:
            # 将句子中的单词表示成index
            encoded_inputs = self.tokenizer(sentence, max_length=self.sen_len, padding='max_length', truncation=True, return_tensors='pt')
            sentence_list.append(encoded_inputs['input_ids'].squeeze(0))  # 将张量转换成列表
        return torch.stack(sentence_list)

    def pad_sequence(self, sentence):
        # BERT tokenizer已处理填充，这里不再需要额外操作
        return sentence

    def labels2tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)    


#外置嵌入层：
class DataPreprocess_w2v:
    def __init__(self, sentences, sen_len, w2v_path):
        self.sentences = sentences  # 句子列表
        self.sen_len = sen_len      # 句子的最大长度
        self.w2v_path = w2v_path    # word2vec模型路径
        self.index2word = []        # 实现index到word转换
        self.word2index = {}        # 实现word到index转换
        self.embedding_matrix = []

        # load word2vec.model
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def make_embedding(self):
        # 为model里面的单词构造word2index, index2word 和 embedding
        for i, word in enumerate(self.embedding.wv.vocab):
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        self.embedding_matrix = np.array(self.embedding_matrix)
        self.embedding_matrix = torch.tensor(self.embedding_matrix)

        # 将"<PAD>"和"<UNK>"加进embedding里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))

        return self.embedding_matrix

    def add_embedding(self, word):
        # 将新词添加进embedding中
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def slice_sequence(self, sequence, window_size, stride):
        """
        使用滑动窗口对序列进行分片。
        :param sequence: 输入序列 (list)
        :param window_size: 滑动窗口大小 (int)
        :param stride: 滑动步长 (int)
        :return: 分片后的序列 (list of lists)
        """
        slices = []
        pad_value = self.word2index.get("<PAD>", 0)  # 使用 <PAD> 的索引值进行填充，如果不存在则默认值为0
        for i in range(0, len(sequence) - window_size + 1, stride):
            slices.append(sequence[i:i + window_size])

        if len(sequence) % stride != 0:  # 如果还有剩余部分，补一个窗口
            last_slice = sequence[-window_size:]
            if len(last_slice) < window_size:
                last_slice = last_slice + [pad_value] * (window_size - len(last_slice))  # 填充到 window_size 长度
            slices.append(last_slice)

        return slices


    def sentence_to_embedding(self, window_size, stride):
        """
        将句子分片并转换为嵌入表示，对分片取平均作为句子的最终表示。
        :param window_size: 滑动窗口大小
        :param stride: 滑动步长
        :return: 嵌入后的句子表示 (numpy array)
        """
        sentence_embedding_list = []
        for sentence in self.sentences:
            # 将句子中的单词转为索引
            sentence_idx = [self.word2index.get(word, self.word2index["<UNK>"]) for word in sentence.split()]
            # 滑动窗口分片
            sliced_sequences = self.slice_sequence(sentence_idx, window_size, stride)            
            feature = []
            for slice_seq in sliced_sequences:
                slice_feature = []
                for idx in slice_seq:
                    slice_feature.append(self.embedding_matrix[idx].numpy())
                # 将每个片段转换为 window_size * embedding_dim 的二维向量
                slice_feature = np.stack(slice_feature)
                feature.append(slice_feature)
            
            # 对所有片段求平均作为句子表示
            feature = np.stack(feature)  # 转换为 (num_slices, window_size, embedding_dim)
            embeddings = np.mean(feature, axis=0)  
            # 对所有分片求平均作为句子表示
            sentence_embedding_list.append(embeddings)

        return torch.Tensor(sentence_embedding_list)
    
    def sentence_word2idx(self):
        sentence_list = []
        for i, sentence in enumerate(self.sentences):
            # 将句子中的单词表示成index
            sentence_index = []
            for word in sentence:
                if word in self.word2index.keys():
                    # 如果单词在字典中则直接读取index
                    sentence_index.append(self.word2index[word])
                else:
                    # 否则赋予<UNK>
                    sentence_index.append(self.word2index["<UNK>"])

            # 统一句子长度
            #sentence_index = self.pad_sequence(sentence_index)
            #分片
            #sentence_index = self.slice_sequence(sentence_index,256)
            sentence_list.append(sentence_index)

        #return torch.LongTensor(sentence_list)
        return sentence_list
    def labels2tensor(self, y):
        y = [int(label) for label in y]

        return torch.LongTensor(y)
    
    
    
    
