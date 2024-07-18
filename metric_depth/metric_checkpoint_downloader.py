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
    parser.add_argument("--environment", "-e", choices=["indoor", "outdoor", "i", "o"], 
                        required=True, help="Specify the environment type for the model (indoor/i or outdoor/o).")
    args = parser.parse_args()
    
    size_mapping = {
        "l": "large",
        "s": "small",
        "b": "base"
    }
    environment_mapping = {
        "i": "indoor",
        "o": "outdoor"
    }

    normalized_size = size_mapping.get(args.size, args.size)
    normalized_environment = environment_mapping.get(args.environment, args.environment)

    urls = {
        ("indoor", "large"): "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true",
        ("indoor", "small"): "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true",
        ("indoor", "base"): "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true",
        ("outdoor", "large"): "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true",
        ("outdoor", "small"): "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true",
        ("outdoor", "base"): "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth?download=true"
    }

    url = urls[(normalized_environment, normalized_size)]
    local_filename = os.path.join("checkpoints", url.split('/')[-1].split("?")[0])
    
    os.makedirs("checkpoints", exist_ok=True)
    download_file(url, local_filename)

if __name__ == "__main__":
    main()
