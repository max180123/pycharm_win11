__all__ = ["test"]  # from 模块 import * 仅仅导入test
def test(a, b):
    print(a+b)


def test1(a, b):
    print(a-b)

if __name__ == '__main__':  # main内置变量（“__main__”） if判断为True运行
    test(1, 2)