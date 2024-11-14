# for_遍历
# name = "maxueqiang"
# for x in name:
#     print(x, end='')

# # 作业 统计字符中有多少个a
# str_ = "asadadasdadahjkakwdsasd"
# count_a = 0
# for num in str_:
#     if num == "a":
#         count_a += 1
# print(f"字符中a的数量：{count_a}")


# # range(num1)-生成[0,num1)的数字序列
# # range(num1,num2,step)-生成[num1,num2)等差为step的数列
# for i in range(10):
#     print(i, end=' ')
# print()
# for i in range(1, 40, 4):
#     print(i, end=' ')


# # 作业求1-100中有多少个偶数
# count_o = 0
# for i in range(1, 100):
#     if i % 2 == 0:
#         count_o += 1
# print("1-100中有：%d个偶数" % count_o)

# for的作用域 for i in range： i作用于for循环 不允许在外使用 i为局部变量


# # 作业 坚持送花100天 每天送花十朵
# i = 1
# for i in range(1, 101):
#     print(f"今天是送花的第{i}天\t", end='')
#     for j in range(1, 11):
#         print("送给小妹的第%d朵花\t" % j, end='')
#     print()
#     print(f"第{i}天送花结束")
# print("坚持%d天送花" % i)

# # 作业9*9 乘法表
# for i in range(1, 10):
#     for j in range(1, i + 1):
#         print(f"{j}*{i}={i * j}\t", end='')
#     print()
#
# for i in range(1, 10):
#     for j in range(1, 11 - i):
#         print(f"{j}*{10 - i}={j * (10 - i)}\t", end='')
#     print()


# # continue 中断所在循环下方语句不执行进行下一次循环 临时中断
# for i in range(1, 6):
#     print("语句1")
#     for j in range(1, 6):
#         print("语句2")
#         continue
#         print("语句3")
#     print("语句4")
#

#  # break 中断循环 永久中断
# for i in range(1, 6):
#     print("语句1")
#     for j in range(1, 6):
#         print("语句2")
#         break
#         print("语句3")
#     print("语句4")
#


# # 作业 发工资
# import random
# account_balance = 10000
# for i in range(1, 21):
#     performance = random.randint(1, 11)
#     if performance < 5:
#         print(f"员工{i}，绩效分{performance}，低于5，不发工资，下一位")
#         continue
#     if account_balance == 0:
#         print("工资发放完了。下个月再来领取吧")
#         break
#
#     account_balance -= 1000
#     print(f"向员工{i}发放工资1000元，账户还剩余{account_balance}元")

import random
account_balance = 10000
for i in range(1, 21):
    performance = random.randint(1, 11)
    if performance < 5:
        print(f"员工{i}，绩效分{performance}，低于5，不发工资，下一位")
        continue
    if account_balance >= 1000:
        account_balance -= 1000
        print(f"向员工{i}发放工资1000元，账户余额{account_balance}元")
    else:
        print("工资发完了，请下个月来领取")
        break


























