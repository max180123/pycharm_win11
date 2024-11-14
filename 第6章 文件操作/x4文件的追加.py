with open("test.txt", "a") as file:
    file.write("\n洗吧罗马")

with open("test.txt", "r") as file1:
    print(file1.read())