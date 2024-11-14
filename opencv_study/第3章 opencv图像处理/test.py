# class student:
#
#     def __init__(self, name, age, address):
#         self.name = name
#         self.age = age
#         self.__address = address
#
#     def __is_orin(self):
#         return self.__address
#
#     def call_by_5G(self):
#         return self.name
#
#     def __str__(self):
#         return f'名字：{self.name}，年龄：{self.age}'
#
#     def __lt__(self, other):
#         return self.age < other.age
#
#
# stu1 = student('张三', 15, '北京')
# stu2 = student('张三', 13, '上海')
# print(stu1 < stu2)
# print(stu1)

class Phone:

    def __init__(self, idmei, produce):
        self.idmei = idmei
        self.produce = produce

    def __str__(self):
        return f'id:{self.idmei},produce:{self.produce}'


huawei = Phone(155, '华为')

print(huawei)
