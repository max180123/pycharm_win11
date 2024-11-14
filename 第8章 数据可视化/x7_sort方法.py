# 列表.sort（key=选择排序依据的函数, reverse = False|True(降序))
# 参数key，要求传入一个参数，表示将列表中的每一个元素都传入函数中，返回排序依据
my_list = [["a", 15], ["b", 17], ["c", 100]]
my_list1 = [["a", 105], ["b", 147], ["c", 1500]]


# 定义排序方法
def choose_sort_key(element):
    return element[1]


# 将元素传入choose_sort_key函数中，决定按照谁来排序
my_list.sort(key=choose_sort_key, reverse=True)
print(my_list)

# 使用lambda方法
my_list1.sort(key=lambda element: element[1], reverse=True)
print(my_list1)
