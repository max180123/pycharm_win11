# 集合：不支持元素的重复 无序（无下标索引）
# 集合={元素1，元素2，元素3....}
# 集合=set（）
my_set = {'hello', 'world', 'python'}

# 元素的添加 集合.add（元素）
my_set.add('why')
print(my_set)

# 元素的移除 集合.remove（元素）
my_set.remove('python')
print(my_set)

# 随机取一个元素
element = my_set.pop()  # 取出原集合元素
print(f"取出的元素是{element}, 取出后的集合：{my_set}")

# 清空集合 集合.clear（）


# 差集 新集合 = 集合1.difference（集合2）
# 取出集合1中有而集合2中无的部分
set1 = {1, 2, 3}
set2 = {1, 5, 6}
set3 = set1.difference(set2)
print(f"set1和set2的差集：{set3}")


# 消除两个集合的差集 集合1.difference_update（集合2）
# 在集合1中，删除和集合2相同的元素
set1.difference_update(set2)
print(f"消除集合1和集合2的差集：{set1}")


# 集合的合并 新集合 = 集合1.union（集合2）
set4 = set1.union(set2)
print(f"set1和set2的合集{set4}")


# 集合的长度 len（集合）
c = len(set4)

# 集合的遍历  不可使用while（无法用下标索引）
for i in set4:
    print(i, end=' ')
print()

# 作业
my_list = ['s', 's', 4, 4, 4, 5, 6]
task = set()
for i in my_list:
    task.add(i)
print(f"去重后得到的集合：{task}")

my_Str = "asdhakjhdwuiaswahkj"
for i in my_Str:
    task.add(i)
print(task)