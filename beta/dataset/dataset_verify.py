import os
from PIL import Image

def check_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(file_path)
            img.verify()
        except (IOError, SyntaxError) as e:
            print(f"flie '{filename}' damage or unsupport: {e}")
        except Exception as e:
            print(f"check '{filename}' error: {e}")

if __name__ == "__main__":
    image_folder = "./train"
    check_images(image_folder)