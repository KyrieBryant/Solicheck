from solidity_parser import parser
import json
import regex as re
import os
import datetime
def process(solfile,output,subdir,filename):
    # 解析Solidity文件（假设文件为0.sol）
    #solfile = r'./contract_func\test_contract\0.sol'
    prefix = filename[:-4]
    
    ast = parser.parse_file(solfile)
    # 读取Solidity源文件内容
    with open(solfile, 'r', encoding='utf-8') as file:
        source_code = file.read()
    # 存储所有函数及其解析树语句
    functions = []

    # 遍历AST并查找函数定义
    # 直接通过属性访问，而不是使用 get() 方法
    for child in ast.children:
        if child['type'] == 'ContractDefinition':
            for subNodes in child.subNodes :
                
                if subNodes.type == 'FunctionDefinition':
                    function = {}
                    if len(subNodes.name)<50:
                        function['name'] = subNodes.name
                        function['parameters'] = subNodes.parameters
                        # 获取函数体（语句）
                        body = subNodes.body if hasattr(subNodes, 'body') else {}                
                        # 存储函数的所有语句
                        function['body'] = body
                        if len(function['body'])==0:
                            continue
                        # 获取函数源代码
                        # 使用正则表达式查找函数定义及其源代码
                        #pattern = r'function\s*'+r'\b' + re.escape(function['name']) + r'\s*\(.*?\)\s*[^;](public|internal|private|external|view|pure)?\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
                        pattern = r'function\s*' + re.escape(function['name']) + r'\s*\(.*?\)\s*.*?\{(?:[^{}]|\{(?:[^{}]*\{[^{}]*\}[^{}]*\})*[^{}]*\})*\}'

                        match = re.search(pattern, source_code, re.DOTALL)
                        
                        if match:
                            #print(match.group(0))
                            function['source_code'] = match.group(0)
                        else:
                            function['source_code'] = ''
                            continue
                        
                        # 添加函数到函数列表
                        functions.append(function)
    os.makedirs(os.path.join(output,subdir),exist_ok=True)
    # 将函数信息写入0.txt文件
    with open(os.path.join(output, subdir,prefix+".txt"), 'w', encoding='utf-8') as f:
        for func in functions:
            f.write(f'"function": "{func["name"]}",\n')
            f.write(f'"source code": "{func["source_code"]}",\n')
            f.write(f'"ast": "{str(func["body"])}",\n')
            f.write("\n")  # 分隔符，方便阅读

    print(f"Functions have been written to {prefix}.txt")


def find(base_folder):
    for label, subdir in enumerate(['undependency', 'dependency']):
        folder_path = os.path.join(base_folder, subdir,'nocomments')
        for filename in os.listdir(folder_path):
            if filename.endswith(".sol"):
                sol_path = os.path.join(folder_path,filename)
                try:
                    process(sol_path,output,subdir,filename)
                except Exception as e:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
                    error_message = f'{timestamp}:error:{sol_path}: {e}\n'
                    print(error_message)
                    with open('newdata_process_error.txt', 'a') as file: file.write(error_message)
                    continue          

tp = 'loop'                
output = rf'.\dataset_func/{tp}'
basefolder = rf'.\dataset\{tp}'

find(basefolder)
#for Dataset3 and Dataset2,from sol to txt 