name = "微博"
stock_price = 19.9
stock_code = 114501
stock_price_increase_price = 1.2
err_count = 0
while True:
    increase_day = input("请输入增长天数")
    try:
        increase_day = int(increase_day)
        break
    except ValueError:
        err_count += 1
        if err_count >= 3:
            print("输入次数过多，退出程序")
            exit()
        print("请输入整数")
increase_days = stock_price * stock_price_increase_price ** increase_day
print(f"微博的股价为{stock_price}元，股票代码为{stock_code}，增长{increase_day}天后，股价为{increase_days}元")
print("微博的股价为%.2f元" % increase_days)