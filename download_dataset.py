from pathlib import Path
import tarfile
import requests

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive_path = Path("temp/flower_photos.tgz")
data_dir = Path("data/trainval")

# 1. Скачивание
if not archive_path.exists():
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    print("Скачивание датасета...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with archive_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

# 2. Распаковка
if not (data_dir / "roses").exists():
    print("Распаковка изображений...")
    data_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if member.name.startswith("flower_photos/") and member.isfile():
                member.name = member.name.removeprefix("flower_photos/")
                tar.extract(member, path=data_dir, filter="data")

    print("Распаковка завершена.")
