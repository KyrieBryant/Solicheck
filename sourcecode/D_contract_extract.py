import pandas as pd
import os
# 本文件旨在从复合合约中提取没有bug的子合约，并且记录标签
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

def non_bug_extract(excel_file, sheet_name, contract_directory, type_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    bug_contract = set()
    all_contract = set()
    notfound = set()
    # 创建用于存放标签的文本文件
    label_file = os.path.join(contract_directory, f'{type_name}_label.txt')
    with open(label_file, 'w') as label_txt:
        label_txt.write("Filename,Ground Truth\n")
    
    # 遍历csv文件
    
    for index, row in df.iterrows():
        ####记录所有sol名称
        contract_name = row['contract']
        filename = row['file']
        all_contract.add(str(filename)+'.sol')
        # 如果子合约没有漏洞,重命名导出
        file_path = os.path.join(contract_directory, type_name+'/'+str(filename)+'.sol')
        #检查有没有这个文件
        if not os.path.exists(file_path):
            notfound.add(str(row['file'])+'.sol')
            continue
        #有这个文件就继续执行
        if row['ground truth'] == 0:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                start_index = text.find(f'contract {contract_name}')
                if start_index != -1:
                    end_index = find_matching_brace(text, start_index)
                    if end_index != -1:
                        contract_content = text[start_index:end_index+1]

                        # 导出子合约为新的sol文件，并更改名称
                        output_file_path = os.path.join(contract_directory, type_name+'/'+str(filename*100000+index+1)+ '.sol')
                        with open(output_file_path, 'w',encoding='utf-8') as sol:
                            sol.write(contract_content)

                         # 将子合约标签信息写入文本文件
                         # 乘以100000只是命名，表示是从复合合约中提取的子合约
                        with open(label_file, 'a') as label_txt:
                            label_txt.write(f"{str(filename*100000+index+1)}.sol,{row['ground truth']}\n")

        else:
            #记录有漏洞的母sol
            #写母合约标签为1的信息
            if str(filename)+'.sol' not in bug_contract:
                bug_contract.add(str(row['file'])+'.sol')
                with open(label_file, 'a') as label_txt:
                    label_txt.write(f"{str(filename)}.sol,{row['ground truth']}\n")
    
    #写母合约标签为0的信息：
    nomal_contract = list(all_contract - bug_contract-notfound)
    with open(label_file, 'a') as label_txt:
        for sol_name in nomal_contract:
            label_txt.write(f"{sol_name},0\n")
            
    print(f'type:{type_name} {notfound}  {len(notfound)} files not founded  all:{len(all_contract)} bug:{len(bug_contract)}')                           
def main():
    excel_file = r'C:\A_github\Clear-main\Dataset\ground truth label.csv'
    contract_directory = r'C:\A_github\Clear-main\Dataset\Clean_Data'
    sheets = [
              'reentrancy',
              'block number dependency',
              'dangerous delegatecall',
              'ether frozen',
              'ether strict equality',
              'timestamp dependency',
              'unchecked external call'
              ]
    type = ['RE','BN','DE','EF','SE','TD','UC']

    index = 0
    for i in sheets:
        type_name = type[index]
        index += 1
        sheet_name = i
        non_bug_extract(excel_file, sheet_name, contract_directory, type_name)

if __name__ == "__main__":
    main()
