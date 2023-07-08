import os

# Go in the directory FRAMES then rename all the files in the directory removing the prefix '$Frame'
print(os.getcwd())

# get all the files under the directory and subdirectories
def rename(dir):
    print("Current directory: ", os.getcwd())
    os.chdir(dir)
    
    for filename in os.listdir():
        if os.path.isdir(filename):
            rename(filename)
        else:
            if filename.startswith('$Frame'):
                new_filename = filename[6:]
                os.rename(filename, new_filename)



rename('./src/FRAMES/')