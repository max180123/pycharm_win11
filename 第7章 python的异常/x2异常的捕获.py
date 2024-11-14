# 对可能出现的bug提前准备，提前处理
# try:
#     可能发生错误的代码
# except:
#     如果出现异常执行的代码

# 基本捕获异常(捕获全部异常）
try:
    file = open("linux.txt", "r", encoding="UTF-8")
except:
    file = open("linux.txt", "w")
finally:
    file.close()

# 捕获特定异常
try:
    print(name)
except NameError as e:
    print(e)
    print("出现了变量未定义的异常")

# 捕获多个异常
try:
    print(name)
    print(1/0)
except (NameError, ZeroDivisionError) as e:
    print("变量，除0出现异常")

# 捕获全部异常
try:
    print(name)
except Exception as e:  # Exception 顶级异常
    print("异常")
else:  # 未出现异常执行的语句
    print("未出现异常")
finally:  # 无论如何都执行的代码  一般用于资源关闭
    print("hh")
