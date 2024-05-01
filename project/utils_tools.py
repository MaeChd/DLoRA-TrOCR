import pandas as pd



def list2csv(data_list, cols, save_dir):
    # 创建 DataFrame 对象
    df = pd.DataFrame(data_list, columns=cols)
    # 保存到 CSV 文件
    df.to_csv(save_dir, index=False)

def dict2csv(dicts,save_dir):
    df = pd.DataFrame(dicts)
    df.to_csv(save_dir,index=False)
