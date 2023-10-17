import os, json

def count_files_in_dir(dir):
    list=[]
    for sub_name in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, sub_name)):
            path = os.path.join(dir, sub_name)
            files = os.listdir(path)
            list.append((path, len(files)))

    list.sort(key=lambda x:x[1], reverse=False)
    return list

if __name__=="__main__":
    import sys
    dir = sys.argv[1]
    #dir = '../Genshin4.1_ZH/Chinese/'
    list = count_files_in_dir(dir)
    print(json.dumps(list, indent=2, ensure_ascii=False))