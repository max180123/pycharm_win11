import random

# 定义银行账户类
class BankAccount:
    # 初始化账户，设置初始余额和用户姓名
    def __init__(self, name, initial_balance=5000000):
        self.name = name
        self.balance = initial_balance

    # 显示用户操作菜单
    def display_menu(self):
        print(f"{self.name}您好！欢迎来到银行，请选择操作：")
        print("查询余额 [输入1]")
        print("存款 [输入2]")
        print("取款 [输入3]")
        print("退出 [输入4]")

    # 查询余额操作
    def check_balance(self):
        print(f"{self.name}，您好！您账户的余额为{self.balance}元")

    # 存款操作
    def deposit(self):
        amount = int(input("请输入存款金额："))
        self.balance += amount
        print(f"{self.name}，您好！您成功存入{amount}元，您的余额为{self.balance}元")

    # 取款操作
    def withdraw(self):
        amount = int(input("请输入取款金额："))
        if amount > self.balance:
            print("您的余额不足！")
        else:
            self.balance -= amount
            print(f"{self.name}，您好！您取款{amount}元，账户余额：{self.balance}元")

# 主程序入口
def main():
    # 获取用户姓名
    name = input("您好，请输入您的名字：")
    # 创建银行账户对象
    account = BankAccount(name)

    # 主循环，处理用户操作
    while True:
        # 显示菜单
        account.display_menu()
        # 获取用户选择
        choice = input("请输入您的选择：")

        # 根据用户选择执行相应操作
        if choice == '1':
            account.check_balance()
        elif choice == '2':
            account.deposit()
        elif choice == '3':
            account.withdraw()
        elif choice == '4':
            # 提供确认退出选项
            confirm = input("确认退出吗？(输入 'Y' 确认，其他键取消)：")
            if confirm.lower() == 'y':
                break
        else:
            print("无效的选择，请重新输入。")

# 如果脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
