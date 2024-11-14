# 序列：内容连续、有序，可使用下标索引的数据容器
# 列表、元组、字符串
# 序列切片：new = 序列[起始：结束:步长]
my_list = list()
for i in range(10):
    my_list.append(i)

my_str = "i love you"
new_list1 = my_list[::2]  # 以2为步长从头开始取
ner_str = my_str[::-1]
new_list2 = my_list[-1:-3:-1]  # 从后取三个元素
print(new_list1)
print(new_list2)
print(ner_str)



# 作业
task = "学python，来黑马程序员，月薪过万"
task = task[::-1]
# 得到”黑马程序员“
black_value = task.index("黑")
ee_value = task.index("员")
task_out = task[black_value:ee_value-1:-1]  # 逆序取出黑马程序员

# 方式二
str_list = task.split("，")
str_list1 = str_list[1].replace('来', '')[::-1]
print(str_list1)