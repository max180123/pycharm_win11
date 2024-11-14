def str_reverse(s):
    print(s[::-1])  # 将字符串反转返回  [起始：终止：步长]


def substr(s, x, y):  # 按照下标x和y，对字符串进行切片
    print(s[x:y:1])


if __name__ == '__main__':
    str_reverse("asdfghjkl")
    substr("asdfghjkl", 0, 2)