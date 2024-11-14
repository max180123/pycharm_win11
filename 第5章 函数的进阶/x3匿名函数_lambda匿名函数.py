# lambda匿名函数  无名称匿名函数  临时使用一次
# 定义语法：lambda 传入参数：函数体（一行代码）
def test_fuc(abc):
    result = abc(1, 2)
    print(result)


test_fuc(lambda x, y: x * y)
test_fuc(lambda x, y: x ** y)