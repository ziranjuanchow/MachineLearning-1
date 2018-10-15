# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/15 11:05 AM'
'''
Given a 32-bit signed integer, reverse digits of an integer.

Example 1:
Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
'''
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x < 0:
            sign = -1
        else:
            sign = 1
        x = abs(x)
        rst = sign * int(str(x)[::-1])
        return rst if -(2**31)-1 < rst < 2**31 else 0

if __name__ == "__main__":
    solution = Solution()
    x = 120
    print(solution.reverse(x))
