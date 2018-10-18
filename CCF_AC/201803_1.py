# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/18 7:43 PM'

s = input().split(" ")
print(s)
for n in range(s.__len__()):
    s[n] = int(s[n])
result = 0
linshi = 0
for i in s:
    if i == 2:
        if linshi == 0:
            result += 2
            linshi += 2
        else:
            result += linshi + 2
            linshi += 2
    else:
        if i == 1:
            linshi = 0
            result += 1
    if i == 0:
        break
print(result)


