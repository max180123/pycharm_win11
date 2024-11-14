# 异常的传递性
def fuc01():
    print("函数1的开始")
    num = 1/0
    print("函数1的结束")


def fuc02():
    print("函数2的开始")
    fuc01()
    print("函数2的结束")


def main():
    try:
        fuc02()
    except Exception as e:
        print(f"出现的异常：{e}")

main()