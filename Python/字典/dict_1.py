# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/8/15 下午8:17'

'''
字典是由键值对来表示的数据 键-值对称为项
例子：phonebook = {'Alice':'2341','Beth':'9102','Cecil':'3258'}
'''
# 1.可以使用函数 dict从其他映射（如字典）或者键-值对序列创建字典
item = [('name','Gumby'),('age',32)]
d = dict(item)
print(d)

x = {}
x[22] = 'liudong'
print(x)
# 2.一个简单的数据库
'''
一个将人名用作键的字典，每个人都用一个字典来表示
字典包含键'phone'和'addr'，分别与电话号码和地址相关联
'''
people = {
        'Alice':{
            'phone':'2341',
            'addr':'Foo drive 23'
        },
        'Beth':{
            'phone':'9102',
            'addr':'Bar street 42'
        },
        'Cecil':{
            'phone':'3158',
            'addr':'Baz avenue 90'
        }
}
# 电话号码和地址的描述性标签，供打印输出时使用
labels = {
    'phone':'phone number',
    'addr':'address'
}
name = input('Name:')
# 要查找电话号码还是地址
request = input('Phone number(p) or address (a)?')
if request == 'p':
    key = 'phone'
if request == 'a':
    key = 'addr'

if name in people:
    print("{}'s {} is {}.".format(name,labels[key],people[name][key]))