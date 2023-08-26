import requests
from tqdm import tqdm
import os

def download_file_with_progress(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        
        progress_bar.close()

if __name__ == "__main__":

    file_url = "http://region-41.seetacloud.com:10218/jupyter/files/autodl-tmp/dataset.zip" 
    save_path = "dataset.zip"

    download_file_with_progress(file_url, save_path)
    print(f"file has been downloaded to'{save_path}'！")