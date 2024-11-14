money = 5000000
import random


# 定义主界面
def Main_Menu():
    num = input(f"{name},您好！欢迎来到银行，请选择操作：\n查询余额 [输入1]\n存款 [输入2]\n取款 [输入3]\n退出 [输入4]\n请输入您的选择")
    return num


# 查询函数
def check_balance():
    print(f"{name}，您好！您账户的余额为{money}元")


# 存款函数
def savings_account():
    saving = random.randint(5000, 10000)
    global money
    money += saving
    print(f"{name}，您好！您成功存入{saving}元\n您的余额为{money}元")


# 取款函数
def withdraw_money():
    global money
    while True:
        withdraw = int(input("请输入您的取款金额"))
        if withdraw > money:
            print("您的余额不足！")
            continue
        else:
            money -= withdraw
            print(f"{name},您好！您取款{withdraw}元，账户余额：{money}元")
            break


name = input("您好，请输入您的名字")
while True:
    num = Main_Menu()
    if num == "1":
        check_balance()
        continue
    elif num == "2":
        savings_account()
        continue
    elif num == "3":
        withdraw_money()
        continue
    else:
        break
