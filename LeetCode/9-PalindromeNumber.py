# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/15 2:04 PM'
'''
Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

Example 1:

Input: 121
Output: true
Example 2:

Input: -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
'''
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        sign = [1,-1][ x< 0]
        if sign == -1:
            return False
        revers = sign * int(str(x)[::-1])
        if revers == x:
            return True
        else:
            return False


if __name__ == "__main__":
    solution = Solution()
    print(solution.isPalindrome(-121))
