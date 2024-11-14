# 函数作为参数传递
# 计算逻辑的传递
def test_fuc(abc):
    result = abc(1, 2)
    print(result)


def summ(x, y):
    return x + y


test_fuc(summ)  # 函数summ作为参数传入test_fuc  传入计算逻辑


def subtract(x, y):
    return x - y


test_fuc(subtract)
