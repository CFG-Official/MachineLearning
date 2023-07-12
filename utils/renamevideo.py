import shutil
import os

folder = "C:\\Users\crist\Downloads\TEST_SET"

base = "Test_Video"
counter = 0
for filename in os.listdir(folder):
    # Reanme the video file
    if filename.endswith(".mp4"):
        os.rename(os.path.join(folder, filename), os.path.join(folder, base + str(counter) + ".mp4"))
        counter += 1