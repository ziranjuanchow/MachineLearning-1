# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/17 11:13 AM'
'''
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.

Example 1:

Input: "()"
Output: true
Example 2:

Input: "()[]{}"
Output: true
Example 3:

Input: "(]"
Output: false
Example 4:

Input: "([)]"
Output: false
Example 5:

Input: "{[]}"
Output: true
'''
# 构建一个栈的形式进行处理
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        mapping = {")":"(", "}":"{","]":"["}
        for i in s:
            if i in mapping:
                top_element = stack.pop() if stack else "#"
                if mapping[i] != top_element:
                    return False
            else:
                stack.append(i)
        return not stack

if __name__  == "__main__":
    solution = Solution()
    s = "()[]{}"
    result = solution.isValid(s)
    print(result)