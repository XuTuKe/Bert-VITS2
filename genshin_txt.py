import os

out_file = f"filelists/genshin_out.txt"
def process():    
    with open(out_file,'w' , encoding="Utf-8") as wf:            
        ch_language = 'ZH'
        for ch_name in os.listdir(f"./raw"):
            path = f"./raw/{ch_name}"
            files = os.listdir(path)
            for f in files:
                if f.endswith(".lab"):
                    with open(os.path.join(path,f),'r', encoding="utf-8") as perFile:
                        line = perFile.readline() 
                        result = f"./dataset/{ch_name}/{f.split('.')[0]}.wav|{ch_name}|{ch_language}|{line}"
                        wf.write(f"{result}\n")

if __name__ == "__main__":
    process()