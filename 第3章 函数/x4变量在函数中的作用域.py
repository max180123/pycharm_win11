# 变量作用域：变量作用的范围
# 局部变量：在函数内部，临时保存数据，当函数调用完后，则销毁局部变量
# def testa():
#     num = 100
#     print(num)


# testa()
# print(num)


# 全局变量：在函数内外都可以生效的变量
# global 将函数内的声明变量为全局变量
num = 100


def testb():
    print(num)

def testc():
    global num  # 局部变量成全局变量
    num = 200  # 局部变量
    print(num)


testb()
testc()
print(num)
