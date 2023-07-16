import shutil
import os

folder = "C:\\Users\crist\Desktop\TMP"

base = "Test_Video"
in_ext = ".txt"
out_ext = ".rtf"
counter = 150
for filename in os.listdir(folder):
    # Reanme the video file
    if filename.endswith(in_ext):
        os.rename(os.path.join(folder, filename), os.path.join(folder, base + str(counter) + out_ext))
        counter += 1