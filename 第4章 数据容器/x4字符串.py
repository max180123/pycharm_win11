# 字符串是字符的容器 一个字符串可以存储任意数量的字符
# 不可修改
my_str = 'i love you'
print(my_str[0])
print(f"love在字符串中的位置：{my_str.index('love')}")  #索引

# 字符串替换 new_str = 字符串.replace（字符串1，字符串2） 将字符串1换为2
new_str = my_str.replace("you", "her")   # 将my_str中的you换成her
print(new_str)


# 字符串的分割 list = 字符串.split（分割符字符串）
new_list = new_str.split(" ")
print(new_list)


# nwe_str = 字符串.strip()  去除首位空格

print(f"字符串{my_str}中e出现的次数：{my_str.count('e')}")
print(f"字符串{my_str}的长度：{len(my_str)}")


t1 = '6 5 5 5 5 '
t1 = t1.replace('5', '6')  # 全部替换
print(t1)



# 作业
task = "itheima itcast boxuegu"
print(f"字符串{task}中it的数量：{task.count('it')}")
task = task.replace(' ', '|')  # 更换字符串的内容
print(task)
task_list = task.split('|')   # 按照|分隔符将字符串转换成列表
print(task_list)