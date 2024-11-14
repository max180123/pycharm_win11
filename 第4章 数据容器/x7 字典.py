# 字典：key对应value  键值对  用key索引  不可重复
# 字典 = {key：value，key：value}
# 字典 = dict（）
my_dict = {'张三': 144, '李四': 150, '王华': {'数学': 144, '语文': 95, '英语': 135}}
my_dict1 = {}  # 集合不可使用{}

print(f"张三的成绩：{my_dict['张三']}")
print(f"王华的数学成绩：{my_dict['王华']['数学']}")


# 新增、修改元素  字典[key] = value
# 存在key 修改  不存在key 新增
my_dict['李四'] = 149  # 将李四的成绩改为149
my_dict['王丽'] = 144  # 新增王丽的成绩144
print(my_dict)


# 删除元素  value = 字典.pop（‘key’）
score = my_dict.pop('张三')
my_dict.pop('李四')
print(f"删除后的字典：{my_dict}，张三的成绩：{score}")


# 清空元素 字典.clear（）


# 获取字典全部的key  keys = 字典.key（）
keys = my_dict.keys()
for i in keys:
    print(f"字典的key是：{i}\t")
    print(f"字典的value是：{my_dict[i]}\t")

for i in my_dict:
    print(f"字典的key：{i}")
    print(f"字典的value：{my_dict[i]}")

