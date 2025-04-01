from gensim.models import word2vec
import os
from utils import read_file,read_contract,read_morecontract
from gensim.models.fasttext import FastText
import os
import re

def train_word2vec(x):
    """
    LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
    vector_size：是每个词的向量维度；size
    window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词；
    min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃；
    sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
    epochs (int, optional) – 迭代次数，默认为5,iter
    :param x: 处理好的数据集
    :return: 训练好的模型
    """
    return word2vec.Word2Vec(x, size=768, window=5, min_count=5, sg=1, iter=10)


# 读取文本文件
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

# 预处理文本（去除特殊符号，进行分词）
def preprocess_text(lines):
    processed_lines = []
    for line in lines:
        # 去除特殊符号，只保留字母和数字
        line = re.sub(r'[^A-Za-z0-9\s]', ' ', line)
        # 分词并转换为小写
        words = line.lower().split()
        processed_lines.append(words)
    return processed_lines


if __name__ == '__main__':
    #data_dir = './contract/RE'
    root_dir = f'./dataset_func/'
    model_dir = './model/newdataset'
    if not os.path.exists(root_dir):os.makedirs(root_dir)
    if not os.path.exists(model_dir):os.makedirs(model_dir)
#w2v:
    train_x = read_morecontract(root_dir)
    print("training text data and transforming to vectors by skip-gram...")
    model = train_word2vec(train_x)
    print("saving w2v model...")
    model.save(os.path.join(model_dir, 'w2v_new_dataset_768.model'))
    
    
    # fastext
    # 创建临时文件并写入训练数据 
    print("train FastText model...")
    with open('./temp_train_data.txt', 'w', encoding='utf-8') as f: 
        for line in train_x: 
            f.writelines(line)
            f.write('\n')
    #训练FastText模型 
    file_path = './temp_train_data.txt'
    lines = read_file(file_path)
    processed_lines = preprocess_text(lines)

    # 输出部分预处理后的文本，进行检查
    print(processed_lines[:5])

    model = FastText(sentences=processed_lines, size=768, window=5, min_count=1, workers=4)    #保存模型 
    model.save(os.path.join(model_dir, 'fasttext_model_768.model'))
    
    print("saving FastText model...")
    #加载FastText模型
    # ft_path = "./model/newdataset/fasttext_model_768.model"
    # embedding = FastText.load(ft_path)
    # # 示例单词的向量
    # word_vector = embedding.wv.index2word
    # print(len(word_vector))
