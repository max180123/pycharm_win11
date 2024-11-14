# 列表的定义 list=[元素1,元素2,元素3] noun=list（）  元素类型不受限
list_t = ['李华', 666, True, [2, 3]]
for i in range(4):
    print(list_t[i], type(list_t[i]), end='\t\t')  # 正序取0,1,2.....
print()
for i in range(4):
    print(list_t[-i - 1], type(list_t[i]), end='\t\t')  # 逆序取-1，-2，-3......
print()

# 列表的下标索引 [0,1,2,3]
#             [-4,-3,-2,-1]
print(list_t[3][1])

# 列表的常用操作
# 方法：将函数定义为class（类）的成员，函数被称为方法 list.

# 查找某元素的下标 列表.index()
list_t = ['李华', 666, True, [2, 3]]
print(list_t.index('李华'))
list_t1 = [1, 2, 3]
# 修改元素 插入元素 列表.insert（下标， 元素）
# 追加元素（尾部） 列表.append（元素）
# 追加元素（一批） 列表.extend（数据容器）
# 删除元素 del 列表[下标] 列表.pop（下标）
# 删除元素 列表.remove（元素）
# 清空列表 列表.clear（）
# 统计某元素 列表.count（元素）
# 统计列表元素 len（列表）
list_t[0] = '张三'  # 修改
list_t.insert(1, '插入元素')  # 插入
list_t.append("追加元素")  # 在尾部追加单个元素
list_t.extend(list_t1)  # 在尾部追加一批元素
del list_t[0]  # 删除列表中的第一个元素
list_t2 = list_t.pop(0)  # 取出并删除元素
list_t.remove('追加元素')  # 找到元素并删除第一个元素
print(list_t, '    ', list_t.count(2), len(list_t))


# 列表的遍历
def list_while_func(list_):  # 定义while遍历函数
    index = 0
    while index < len(list_):
        print(list_[index], end='  ')
        index += 1


def list_for_func(list_):  # 定义for遍历函数
    for i in range(0, len(list_)):
        print(list_[i], end='  ')


task = [21, 25, 21, 23, 22, 20]
list_while_func(task)

