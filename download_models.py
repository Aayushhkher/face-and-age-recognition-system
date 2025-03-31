import urllib.request
import os

def download_file(url, filename):
    """Download a file from a URL."""
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # URLs for the model files
    urls = {
        'age_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net_deploy.prototxt',
        'age_net.caffemodel': 'https://drive.google.com/uc?export=download&id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW'
    }
    
    # Download each file
    for filename, url in urls.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    main() 