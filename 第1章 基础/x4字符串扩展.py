# 字符串定义 单引号'' 双引号"" 三引号""""""""

# 在字符串内包含双引号
name = '"hello world"'
# 转义字符
name1 = "\"hello world\""
print(name, name1)


# 字符串的拼接 + 只能拼接字符串
age = 15
print("我的年龄是：" + str(age) + "岁")


# 字符串格式化 %：我要占位 s：将变量变成字符串放入占位的位置
avg_num = 15.6
avg_salary = 15000
print("学生的平均年龄是%s岁,他们的零花钱是%s元" % (int(avg_num), avg_salary))

# 占位 %s（转为字符串） %d（转为整数） %f（转为浮点数）-精度控制 %m.nf   m 数字宽度（小数点算入） n小数点精度（四舍五入）
# 字符串格式化 快速写法 format 格式化  f“内容{变量}”  不限数据类型 不做精度控制
name2 = "疫苗公司"
setup_year = 2018
stock_price = 19.96
print("公司的名为%s，成立于%d年，股票价格%.1f元" % (name2, setup_year, stock_price))
print(f"公司的名为{name2}，成立于{setup_year}，今日股票价格为{stock_price}元")

# 字符串格式化 对表达式的格式化
print("1*1的结果是：%d" % (1*1))
print(f"1*1的结果是：{1*1}")
print("字符串在Python的类型是%s" % type("字符串"))