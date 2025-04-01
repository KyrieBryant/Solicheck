import os
from solidity_parser import parser
import json
#把sol文件转换为ast，以json保存
def process_sol_file(sol_file_path,typename):#将单个sol转换为ast并以json存储
    file_name = os.path.splitext(os.path.basename(sol_file_path))[0]
    print(f'processing {typename}: {file_name}.sol ...')
    with open('errors.txt','a') as errorlog:
    # 解析.sol文件的AST
        try:
            ast = parser.parse_file(sol_file_path)
    #保存json
            with open(f'json/{typename}/{file_name}.json', 'w', encoding='utf-8') as json_file:
                json.dump(ast, json_file, ensure_ascii=False, indent=4)
            print(f'save {typename}: {file_name}.json ...')
    # 获取ast对象
            ast_dic = parser.objectify(ast)
    # 获取文件名（不含扩展名）的前缀部分    
            #contracts = ast_dic.contracts    # 是字典形式,value是合约对象
            #contracts_name = list(contracts.keys())# 获取sol文件中所有的合约名称
            #print(contracts_name)
    #其实可以不用以json格式保存，可以直接在这步对json处理，然后得到最终的txt预输入数据
    #把两个代码合并一下
    # 写入JSON文件
    # with open(f"contract\{typename}\{file_name}.json", 'w') as json_file:
    #     json_file.write(ast_json)
        
    #                           json->txt    
            save_nested_dict_to_txt(ast_dic._node,'contract',typename,file_name)    
        except Exception as e:
            error_message = f'error:{sol_file_path}: {e}\n'
            print(error_message)
            errorlog.write(error_message)
    #  处理文件夹
def process_contract_folder(folder_path,typename):
    # 遍历文件夹中的.sol文件并处理
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".sol"):#判断是不是sol文件
            #判断是否已经解析过：
            txt_name = file_name[:-4]
            txt_path = r'C:\Users\ASUS\PyTorch_exp\文本分类\LSTM\contract/'+typename+'/'+txt_name+ '.txt'
            json_path = r'C:\Users\ASUS\PyTorch_exp\文本分类\LSTM\json/'+typename+'/'+txt_name+ '.json'
            if os.path.exists(json_path):
                continue
            sol_file_path = os.path.join(folder_path, file_name)
            process_sol_file(sol_file_path,typename)
            

#           json -> txt
def save_nested_dict_to_txt(dictionary, filepath, typename, filename, level=0):
    folder_path = os.path.join(filepath, typename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    with open(os.path.join(folder_path, f"{filename}.txt"), "a") as file:
        for key, value in dictionary.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        save_nested_dict_to_txt(item, filepath, typename, filename, level + 1)
            elif isinstance(value, dict):
                file.write(f"{level} {key}: \n")
                save_nested_dict_to_txt(value, filepath, typename, filename, level + 1)
            else:
                file.write(f"{level} {key}: {value}\n")
  

if __name__ == "__main__":
    #process_sol_file(sol_file_path)    单个测试

    # 设置智能合约源代码文件夹的路径
    contract_folder = r"C:\A_github\Clear-main\Dataset\Clean_Data"
    typename = ['EF','SE','TD','UC','RE','BN','DE','IO']
    #处理每个合约数据文件夹：
    for i in typename:
        contract_path = os.path.join(contract_folder, i)#文件夹路径
        process_contract_folder(contract_path,i)#处理每个文件夹
 
    
