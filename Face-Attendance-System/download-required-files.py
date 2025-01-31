import os
import urllib.request
import bz2

def download_file(url, output_path):
    """Downloads a file from a URL and saves it locally."""
    print(f"Downloading {os.path.basename(output_path)}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded {os.path.basename(output_path)} successfully!")

def extract_bz2(file_path, output_path):
    """Extracts a .bz2 file and saves the extracted content."""
    print(f"Extracting {os.path.basename(file_path)}...")
    with bz2.BZ2File(file_path, 'rb') as fr, open(output_path, 'wb') as fw:
        fw.write(fr.read())
    print(f"Extracted to {output_path} successfully!")
    os.remove(file_path)  # Remove the compressed file after extraction

def main():
    """Main function to download and extract the required files."""
    files = {
        "shape_predictor_68_face_landmarks.dat.bz2": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat.bz2": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }

    for compressed_file, url in files.items():
        extracted_file = compressed_file.replace(".bz2", "")
        if not os.path.exists(extracted_file):
            download_file(url, compressed_file)
            extract_bz2(compressed_file, extracted_file)
        else:
            print(f"{extracted_file} already exists, skipping download.")

if __name__ == "__main__":
    main()
