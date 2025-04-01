import os
#the file is aimed to clean the comments in sol file
def remove_comments(input_file, output_file):
    with open(input_file, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w',encoding='utf-8') as f:
        in_comment_block = False
        for line in lines:
            # 检查是否处于多行注释块内
            if in_comment_block:
                if '*/' in line:
                    in_comment_block = False
                continue
            # 处理单行注释
            if '//' in line:
                line = line[:line.index('//')] + '\n'
            # 处理多行注释
            elif '/*' in line:
                if '*/' in line:
                    line = line[:line.index('/*')] + line[line.index('*/') + 2:]
                else:
                    in_comment_block = True
                    continue
            f.write(line)

def process_directory(directory,type_name):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.sol'):
                input_file = os.path.join(root, file)
                output__root_directory = r'C:\A_github\Clear-main\Dataset\Clean_Data'
                output_directory = os.path.join(output__root_directory, type_name)
                output_file = os.path.join(output_directory, file)
                remove_comments(input_file, output_file)

if __name__ == "__main__":
    RE = r'C:\A_github\Clear-main\Dataset\reentrancy (RE)'
    BN = r'C:\A_github\Clear-main\Dataset\block number dependency (BN)'
    DE = r'C:\A_github\Clear-main\Dataset\dangerous delegatecall (DE)'
    EF = r'C:\A_github\Clear-main\Dataset\ether frozen (EF)'
    IO = r'C:\A_github\Clear-main\Dataset\integer overflow (IO)'
    SE = r'C:\A_github\Clear-main\Dataset\ether strict equality (SE)'
    TD = r'C:\A_github\Clear-main\Dataset\timestamp dependency (TD)'
    UC = r'C:\A_github\Clear-main\Dataset\unchecked external call (UC)'
    list = [RE,BN,DE,EF,SE,TD,UC]
    type = ['RE','BN','DE','EF','SE','TD','UC']

    output_directory = r'C:\A_github\Clear-main\Dataset\Clean_Data'
    index = 0
    for path in list:
        type_name = type[index]
        print(f'Process {type_name} contracts')
        process_directory(path,type_name)
        print(f'Process {type_name} done')
        index += 1

