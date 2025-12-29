import urllib.request
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
# Модель Qwen3-8B-GGUF (Q4_K_M - самый легкий вариант, ~5GB)
MODEL_NAME = "Qwen3-8B-Q4_K_M.gguf"
REPO_ID = "Qwen/Qwen3-8B-GGUF"

MODEL_DIR.mkdir(exist_ok=True)
out_path = MODEL_DIR / MODEL_NAME


def download_with_progress(url: str, filepath: Path):
    """Скачивает файл с отображением прогресса."""
    def reporthook(blocknum, blocksize, totalsize):
        if totalsize > 0:
            percent = min(100, (blocknum * blocksize * 100) // totalsize)
            downloaded = blocknum * blocksize
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = totalsize / (1024 * 1024)
            print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    urllib.request.urlretrieve(url, str(filepath), reporthook)
    print()  # Новая строка после прогресса


def main() -> None:
    if out_path.exists():
        print(f"Model already exists: {out_path}")
        return
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Will be saved to: {out_path}")
    print()
    
    # Пробуем использовать huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading with huggingface_hub...")
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_NAME,
            local_dir=str(MODEL_DIR)
        )
        print(f"Model successfully downloaded: {downloaded_path}")
        return
    except ImportError:
        print("huggingface_hub not installed, trying direct download...")
    except Exception as e:
        print(f"Error with huggingface_hub: {e}")
        print("Trying direct download...")
    
    # Пробуем использовать huggingface-cli
    try:
        import subprocess
        print("Trying to download with huggingface-cli...")
        result = subprocess.run(
            [
                "huggingface-cli", "download",
                REPO_ID,
                MODEL_NAME,
                "--local-dir", str(MODEL_DIR),
                "--local-dir-use-symlinks", "False"
            ],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Model successfully downloaded: {out_path}")
            return
        else:
            print(f"huggingface-cli failed: {result.stderr}")
    except FileNotFoundError:
        print("huggingface-cli not found.")
    except Exception as e:
        print(f"Error with huggingface-cli: {e}")
    
    print("\nAll download methods failed.")
    print("\nTry downloading manually:")
    print(f"1. Install: pip install huggingface-hub")
    print(f"2. Or use: huggingface-cli download {REPO_ID} {MODEL_NAME} --local-dir {MODEL_DIR}")
    print(f"3. Or visit: https://huggingface.co/{REPO_ID} and download manually")
    sys.exit(1)


if __name__ == "__main__":
    main()