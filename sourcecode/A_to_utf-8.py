import os
import codecs

# 定义目录路径
directory = r'C:\A_github\Clear-main\Dataset\integer overflow (IO)'
#directory = r'C:\Users\ASUS\PyTorch_exp\文本分类\LSTM\Clean_Data\BN\bug'
# 遍历目录下的所有文件
for root, dirs, files in os.walk(directory):
    for file in files:
        # 确保文件是.sol文件
        if file.endswith('.sol'):
            file_path = os.path.join(root, file)
            # 打开文件并以二进制读取模式读取内容
            with open(file_path, 'rb') as f:
                content = f.read()
            # 将内容转换为UTF-8编码
            content_utf8 = content.decode('utf-8', 'ignore')
            # 写入UTF-8编码的内容到文件中
            with codecs.open(file_path, 'w', 'utf-8') as f:
                f.write(content_utf8)
            print(f"Converted {file} to UTF-8")
