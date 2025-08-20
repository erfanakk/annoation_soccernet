import os
import shutil

# ---- CONFIG ----
src_folder = "./train"   # Folder where .jpg images are now
dst_folder = "./images"  # Folder where images should be moved

# Create destination folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)

# Loop through all files in source folder
for filename in os.listdir(src_folder):
    if filename.lower().endswith(".jpg"):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")

print("✅ All .jpg images moved successfully.")
