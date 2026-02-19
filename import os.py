import os
for file in os.listdir("."):
    if file.endswith(".txt"):
        print(file)
