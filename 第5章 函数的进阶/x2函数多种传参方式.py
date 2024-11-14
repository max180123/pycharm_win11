# 位置参数：调用函数时根据函数定义的参数位置来传递参数
def user_info(name, age, gender):
    print(f"您的名字是{name}，年龄：{age}，性别：{gender}")


user_info("张三", 25, "男")

# 关键字参数：调用函数时通过“键=值”形式传递参数
user_info(name="李华", age=15, gender="男")  # 顺序可打乱
user_info("李四", gender="女", age=115)


# 缺省参数：定义函数时，为参数设置默认值  默认值写在最后
def user_iforma(name, age, gender="男"):
    print(f"名字：{name}，年龄：{age}，性别：{gender}")


user_iforma("张三", 15)


# 不定长参数：调用函数时不确定参数个数
# 位置传递
def test(*args):  # arge是元组tuple  规范args
    print(args)


test("zhangsan", 15, 23)


# 关键字传递  参数传递 键=值
def test1(**kwargs):  # kwargs是字典  规范kwargs
    print(kwargs)


test1(age=15, name="李华")
