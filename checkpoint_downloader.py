import os
import requests
import argparse
from tqdm import tqdm

def download_file(url, local_filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_filename, 'wb') as file:
            for data in tqdm(response.iter_content(chunk_size=8192), total=total_size // 8192, unit='KB', unit_scale=True):
                file.write(data)
        
        print(f"File downloaded successfully: {local_filename}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def main():
    parser = argparse.ArgumentParser(description="Download checkpoint files.")
    parser.add_argument("--size", "-s", choices=["large", "small", "base", "l", "s", "b"], 
                        default="large", help="Specify the size of the file to download (large/l, small/s, base/b). Default is large.")
    args = parser.parse_args()
    
    urls = {
        "large": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
        "small": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
        "base": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true"
    }
    
    size_mapping = {
        "l": "large",
        "s": "small",
        "b": "base"
    }

    # Normalize the input size to its full form (e.g., "l" to "large")
    normalized_size = size_mapping.get(args.size, args.size)  # Retrieve full form or use the default if mapping not found

    # Get the URL for the specified size
    url = urls[normalized_size]
    
    checkpoints_dir = "checkpoints"
    local_filename = os.path.join(checkpoints_dir, f"depth_anything_v2_vit{normalized_size[0]}.pth")  # Using the first letter of the full size form
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    download_file(url, local_filename)

if __name__ == "__main__":
    main()
