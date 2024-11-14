# num_test = 0
# while num_test < 100:
#     num_test += 1
#     print("这是第：%d次测试" % num_test)

# # 求1+.....+100的和
# num_ = 0
# num_sum = 0
# while num_ < 100:
#     num_ += 1
#     num_sum += num_
# print("1+...+100的和为：%d" % num_sum)

# # 猜数字 不限制次数 记录猜测的次数
# import random
# num = random.randint(0, 100)
# guess_count = 0
# print("下面是猜数字游戏，您需要猜测产生的随机数字，范围为：0-100")
# while True:
#     guess_count += 1
#     guess_num = int(input("请输入数字："))
#     if guess_num > num:
#         print("猜大了！")
#     elif guess_num < num:
#         print("猜小了！")
#     else:
#         print("恭喜您用了%d次就猜对了" % guess_count)
#         break


# # 送花给女朋友
# day = 0
# while day < 10:
#     day += 1
#     print(f"今天是第{day}天，准备给女朋友送花！")
#     flower = 0
#     while flower < 10:
#         flower += 1
#         print(f"第{day}天送女朋友的第{flower}朵花！")
#     print("我喜欢你！")
# print(f"坚持送花到{day}天！")

# # print() 不换行  对其 制表符\t
# print("hello", end=' ')
# print("world")
# print("whshhw\tsssad")
# print("sadwd\tsdj")

i = 0
while i < 9:
    i += 1
    j = 1
    while j <= i:
        print(f"{j}*{i}={i*j}\t", end='')
        j += 1
    print("")

i = 0
while i < 9:
    j = 1
    while j <= 9 - i:
        print(f"{j}*{9 - i}={(9 - i) * j}\t", end='')
        j += 1
    print("")
    i += 1