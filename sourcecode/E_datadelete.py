import os
import pandas as pd

# 删除在目录中但不在csv中的文件
def list_sol_files(directory):
    # 遍历目录下的所有文件
    ll = os.listdir(directory)
    ll.remove('bug')    
    return set(ll)

def delete_files_by_names(directory, file_names):
    # 遍历集合中的每个文件名
    for file_name in file_names:
        # 构建文件的完整路径
        file_path = os.path.join(directory, file_name)
        # 检查文件是否存在，然后删除
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        else:
            print(f"文件不存在: {file_path}")
            
txt = ['BN','DE','EF','RE','SE','TD','UC','IO']

for i in txt:
    txt_name = i+'_label.txt'
    file_path = 'C:\A_github\Clear-main\Dataset\Clean_Data\\'+txt_name
    df = pd.read_csv(file_path, sep=',')
    print(f'            type:{i}')
    #print(df.head())
    # 统计Filename列的个数
    filename_count = df['Filename'].nunique()
    print("标签的sol个数为:", filename_count)
    filename_set = set(df['Filename'])
    directory = 'C:\A_github\Clear-main\Dataset\Clean_Data\\'+i
    file_names = list_sol_files(directory)
    print("目录的sol个数为:", len(file_names))
    delete_set = file_names-filename_set
    print(delete_set,f'目录比标签多{len(delete_set)}个')
    #删除:
    #delete_files_by_names(directory,delete_set)
