import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("PEXELS_API_KEY")
if not API_KEY:
    sys.exit("PEXELS_API_KEY missing from .env (free key: https://www.pexels.com/api/)")
BASE_URL = "https://api.pexels.com/v1/search"

ENVIRONMENTS = {
    "classroom": [
        "school chair isolated",
        "pencil white background",
        "open book desk",
        "school bag isolated",
        "ruler isolated white",
    ],
    "kitchen": [
        "coffee cup isolated white",
        "cooking pot isolated",
        "kitchen knife isolated",
        "fork spoon isolated white",
        "frying pan isolated",
    ],
    "supermarket": [
        "milk bottle isolated white",
        "apple fruit isolated",
        "bread loaf isolated",
        "shopping basket isolated",
        "water bottle isolated white",
    ],
    "fruits": [
        "red apple white background",
        "banana isolated white",
        "orange fruit isolated",
        "strawberry isolated white",
        "grapes isolated white",
    ],
}

IMAGES_PER_QUERY = 3  # 5 queries × 3 = 15 images per environment


def search_photos(query, per_page=5):
    headers = {"Authorization": API_KEY}
    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape",
    }
    response = requests.get(BASE_URL, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json().get("photos", [])


def download_image(url, filepath):
    if os.path.exists(filepath):
        print(f"  [skip] {os.path.basename(filepath)} already exists")
        return False
    response = requests.get(url, timeout=30, stream=True)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def main():
    base_dir = os.path.join("static", "images")

    for env_name, queries in ENVIRONMENTS.items():
        save_dir = os.path.join(base_dir, env_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'='*50}")
        print(f"Environment: {env_name.upper()}")
        print(f"{'='*50}")

        img_index = 1

        for query in queries:
            print(f"\n  Searching: \"{query}\"")
            try:
                photos = search_photos(query, per_page=IMAGES_PER_QUERY)
            except Exception as e:
                print(f"  [error] Search failed: {e}")
                continue

            downloaded = 0
            for photo in photos:
                # Filter: landscape + min width 800px
                width = photo.get("width", 0)
                height = photo.get("height", 0)
                if width < 800 or width <= height:
                    continue

                url = photo["src"]["large"]
                filename = f"{env_name}_{img_index:02d}.jpg"
                filepath = os.path.join(save_dir, filename)

                try:
                    added = download_image(url, filepath)
                    status = "downloaded" if added else "skipped"
                    print(f"  [{status}] {filename} ({width}x{height}) — {photo.get('photographer', 'unknown')}")
                    img_index += 1
                    downloaded += 1
                except Exception as e:
                    print(f"  [error] Failed to download {filename}: {e}")

            if downloaded == 0:
                print(f"  [warn] No suitable landscape photos found for this query")

        total = img_index - 1
        print(f"\n  Total saved for {env_name}: {total} images")

    print(f"\n{'='*50}")
    print("Done!")


if __name__ == "__main__":
    main()
