my_list = [1, 2, 3, 4, 5]
my_tuple = (1, 2, 3, 4, 5)
my_str = "abcdefg"
my_set = {1, 2, 3, 4, 5}
my_dict = {'key1': 1, 'key2': 2, 'key3': 3, 'key4': 4, 'key5': 5}


# len（）元素个数
print(f"列表的元素个数：{len(my_list)}")
print(f"元组的元素个数：{len(my_tuple)}")
print(f"字符串的元素个数：{len(my_str)}")
print(f"集合的元素个数：{len(my_set)}")
print(f"字典的元素个数：{len(my_dict)}")


# max() min() 最大、最小元素
print(f"列表中的最大、最小值：{max(my_list)}、{min(my_list)}")
print(f"字符串中的最大、最小值:{max(my_str)}、{min(my_str)}")
print(f"字典的最大、最小值：{max(my_dict)}、{min(my_dict)}")   # 字典按照key


# 容器通用转换
# 转换为列表 list（容器）
# 转换为字符串 str（容器）
# 转换为元组 tuple（容器）
# 转换为集合 set（容器）
new_set = set(my_str)  # 将字符串转为集合
new_set1 = set(my_dict)  # 将字典转为集合 保留key
print(new_set)
print(new_set1)


# 容器排序 排序后 = sorted(容器， reverse=True]  排序后成列表
sort_list = sorted(my_list)  # 升序
sort_list1 = sorted(my_list, reverse=True)  # 逆序
print(f"列表的正序：{sort_list}")
print(f"列表的逆序：{sort_list1}")


# 字符串的大小  字符有对应的ASCII码
print("abc" > "ABC")
print("张三" > "张四")