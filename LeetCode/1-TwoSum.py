# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/11 8:13 PM'
import time

'''
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
'''

'''  
时间复杂度代表的是算法执行时间与数据规模之间的增长关系 
空间复杂度代表的是算法的存储空间与数据规模之间的增长关系 常见的是 O(1) O(n) O(n^2)  
'''
# 时间复杂度 O(n^2)  空间复杂度 O（n）
def twoSum1(nums,target):
    start_time = time.time()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                end_time = time.time()
                print (end_time - start_time)
                return [i,j]
    return []

# 时间复杂度O(n) 空间复杂度 O(n)
def twoSum2(nums,target):
    start_time = time.time()
    look_up = {}
    for i, num in enumerate(nums):
        print(i,num)
        if target - num in look_up:
            end_time = time.time ()
            print (end_time - start_time)
            return [look_up[target-num], i]
        look_up[num] = i
    return []

numbers = [2, 7, 11, 15]
target = 17
print(twoSum1(numbers,target))
print(twoSum2(numbers,target))
# twoSum1 时间 5.245208740234375e-06 5.0067901611328125e-06
# twoSum2 时间 4.601478576660156e-05 5.888938903808594e-05

