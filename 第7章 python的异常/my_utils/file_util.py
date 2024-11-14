def print_file_info(file_name):  # 接受传入文件的路径（文件名）
    file = None
    try:
        file = open(file_name, "r", encoding="UTF-8")
    except Exception as e:
        print(f"存在异常{e}")
    else:
        print("文件的内容：")
        print(file.read())
    finally:
        if file:  # file的判读避免文件为空时 file.close出现错误
            file.close()


def append_to_file(file_name, date):  # 接受文件路径以及传入数据
        file = open(file_name, "a", encoding="UTF-8")  # 追加用a w会覆盖
        file.write(f"\n{date}")
        print(f"成功将{date}追加入{file_name}中")
        print("追加后的文件内容为：")
        file = open(file_name, "r", encoding="UTF-8")
        print(file.read())
        file.close()


if __name__ == '__main__':
    print_file_info("D:/python_learn/第7章 python的异常/fileutil.txt")
    append_to_file("D:/python_learn/第7章 python的异常/fileutil.txt", "追加内容")

