# ctrl + / 多行注释
# age = 18
# print(f"今年我已经{age}岁了")
# if age >= 18:
#     print("我已经成年了")
#     print("我即将步入大学生活")
# print("不受if判断影响")

# print("欢迎来到黑马儿童游乐场，儿童免费，成人收费")
# age = eval(input("请输入你的年龄"))
# if age >= 18:
#     print("您已成年，游玩需补票10元")
# else:
#     print("您未成年，无需补票")
# print("祝您游玩愉快！")


# print("欢迎来到黑马游乐园")
# height = eval(input("请输入您的身高："))
# vip_leval = eval(input("请输入您的vip等级："))
# if height <= 120 or vip_leval >= 3:
#     print("您可免费游玩")
# # elif vip_leval >= 3:
#     # print("您的vip等级不小于3，可免费游玩")
# else:
#     print("不好意思，麻烦您补票十元")


# # 作业
# import random
# num = random.randint(0, 10)
# guess_count = 0
# print("猜猜心里面的数字0-10，您有5次机会")
# while True:
#     guess_count += 1
#     guess_num = eval(input("请输入数字"))
#     if guess_count == 5:
#         print("次数上限，您未猜出数字为：%d" % num)
#         break
#     elif guess_num > num:
#         print("猜大了")
#     elif guess_num < num:
#         print("猜小了")
#     elif guess_num == num:
#         print("恭喜您猜对了")
#         break

# # 判断语句的嵌套
# print("欢迎来到黑马动物园")
# if eval(input("请输入您的身高（cm）：")) > 120:
#     print("您的身高超过限制需要检验您的vip等级")
#     if eval(input("请输入您的vip等级：")) >= 3:
#         print("您可免费游玩！")
#     else:
#         print("您需要支付10元")
# else:
#     print("欢迎游玩")


# age = int(input("请输入您的年龄："))
# if 18 <= age < 30:
#     if int(input("请输入您的入职时间：")) > 2:
#         print("您可以领取礼物")
#     elif int(input("请输入您的级别：")) > 3:
#         print("您可以领取礼物")
#     else:
#         print("您不可领取礼物")
# else:
#     print("不好意思，您不能领取礼物")


