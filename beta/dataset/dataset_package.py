import zipfile
import os

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    folder_to_zip = "dataset"
    zip_filename = "dataset.zip"
    zip_folder(folder_to_zip, zip_filename)
    print(f"folder '{folder_to_zip}' has been successfully compressed in '{zip_filename}'！")
