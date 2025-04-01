import pandas as pd
import os
#this file is to extract the bug contract from contracts with bug
def find_matching_brace(text, start_index):
    stack = []
    for i in range(start_index, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            stack.pop()
            if len(stack) == 0:
                return i
    return -1        

def bug_extract(excel_file, sheet_name, contract_directory,type_name):
    # 读取 Excel 文件的指定工作表
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # 遍历工作表中的每一行
    for index, row in df.iterrows():
         # 检查文件是否存在
        contract_name = row['contract']  # 获取子合约的名称
        file_name = os.path.join(contract_directory, type_name+'/'+str(row['file'])+'.sol')  # 获取母合约的文件路径
        if not os.path.exists(file_name):
            print(f"type:{type_name}  File  {str(row['file'])+'.sol'}  not found.")
            continue
        
        # 如果合约有漏洞
        if row['ground truth'] == 1:
            # 打开母合约文件
            with open(file_name, 'r',encoding='utf-8') as f:
                text = f.read()  # 读取母合约文件内容

                # 寻找子合约的位置
                start_index = text.find(f'contract {contract_name}')
                if start_index != -1:
                    # 寻找合约内容的结束位置
                    end_index = find_matching_brace(text, start_index)
                    if end_index != -1:
                        # 提取合约内容
                        contract_content = text[start_index:end_index+1]

                        # 将合约内容写入新的 Solidity 文件中，以子合约名称命名
                        with open(os.path.join(contract_directory, type_name+'/bug/'+contract_name + '.sol'), 'w',encoding='utf-8') as sol:
                            sol.write(contract_content)

def main():
    # label file
    excel_file = r'C:\A_github\Clear-main\Dataset\ground truth label.csv'
    # cleaned data root dir
    contract_directory = r'C:\A_github\Clear-main\Dataset\Clean_Data'
    # type name 
    sheets = [
              'reentrancy',
              'block number dependency',
              'dangerous delegatecall',
              'ether frozen',
              'ether strict equality',
              'timestamp dependency',
              'unchecked external call',
              'interger overflow'
              ]
    type = ['RE','BN','DE','EF','SE','TD','UC','IO']

    index = 0
    for i in sheets:
        type_name = type[index]
        index += 1
        sheet_name = i  # Excel 文件中要读取的工作表名称
        bug_extract(excel_file, sheet_name, contract_directory,type_name)

if __name__ == "__main__":
    main()
