# # 函数的返回值：程序中的函数完成后给调用者的结果
# def sum1(x, y):
#     he = x+y
#     return he
# he1 = sum1(4, 5)
# print(f"4+5的和为{sum1(4, 5)}、{he1}")


# # 无return 返回none 空值
# def say_hai():
#     print("你好")
# r = say_hai()
# print(f"返回值：{r}\n返回值类型为{type(r)}")


# # 在if语句中None 等同于False
# age = int(input("年龄："))
# def check_age(age):
#     if age >= 18:
#         return "success"
# result = check_age(age)
# if not result:
#     print("未成年禁止入内！")
#
# # 声明空值
# name = None


def qiuhe(x, y):
    """
    qiuehe函数用于求和
    :param x: 形式参数x是相加的第一个数字
    :param y: 形式参数y是相加的另一个数字
    :return: 返回两个值相加的结果
    """
    result = x+y
    return result
r = qiuhe(4, 5)
print(f"{r}")