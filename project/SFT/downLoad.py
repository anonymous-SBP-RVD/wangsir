# Listing 7.1 Downloading the dataset - Fixed
import json
import os
import urllib.request  # Import urllib.request (NOT urllib.request.request...)

def download_and_load_file(file_path, url):
    # 如果文件不存在，下载文件
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    
    # 读取并解析JSON文件
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
print("Example entry:\n", data[50])