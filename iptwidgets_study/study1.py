import ipywidgets as widgets # 导入widgets库
from IPython.display import display # 导入展示库

# 定义一个计算函数
def f(a, b, c):
    # 创建表达式字符串
    expression = f'{a}{b}{c}'
    # 计算表达式的值
    result = eval(expression)
    print('{}{}{}={:.2f}'.format(a, b, c,result)) # 结果保留两位小数

a = widgets.FloatText(
    min=-100,
    max=100,
    step=0.1,
    description='a',
    read_format='.1f')

a1 = widgets.FloatSlider(
    min=-100,
    max=100,
    step=0.1,
    read_format='.1f')

# 将部件a和a1链接在一起
alink = widgets.jslink((a, 'value'), (a1, 'value'))

b = widgets.Dropdown(
    options=['+', '-', '*', '/'],
    value='+',
    description='算数符')

c = widgets.FloatText(
    min=-100,
    max=100,
    step=0.1,
    description='c',
    read_format='.1f')

c1 = widgets.FloatSlider(
    min=-100,
    max=100,
    step=0.1,
    read_format='.1f')

# 将部件c和c1链接在一起
clink = widgets.jslink((c, 'value'), (c1, 'value'))

# 创建交互式面板
out = widgets.interactive_output(f, {'a':a, 'b':b, 'c':c})
ui = widgets.VBox([widgets.HBox([a, a1]), widgets.HBox([c, c1]), b, out])
display(ui, out)
