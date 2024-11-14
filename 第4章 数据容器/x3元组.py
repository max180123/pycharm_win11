# 元组：一旦定义完成就不可修改 存储多个不同类型的元素
# 定义：变量名称=（元素1，元素2....）
#      变量名称=tuple（）
t1 = (1,)  # 单个元素在最后需加,
tuple_1 = tuple()
print(f"{tuple_1}的类型为{type(tuple_1)}")

t2 = (1, 2, 3, 4, 4, [1, 2])
print(t2[0])  # 索引同列表
print(t2.count(2))  # 查找元素2的位置
print(t2.count(4))  # 统计元素4的数量
print(len(t2))  # 统计元组的元素数量

# 可使用while和for遍历


# 元组内的列表可修改 特例
t2[-1][0] = 0
print(t2)



# 作业
task = ('周杰伦', 11, ['football', 'music'])
print(f'元组task中年龄的下标：{task.index(11)}')
print(f'学生的姓名：{task[0]}')
# del task[2][1]  # 删除
task[2].remove('football')  # 删除football
task[2].insert(1, 'coding')  # 插入coding
task[2].append('coding')
print(task)