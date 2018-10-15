# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/15 9:59 AM'
'''
Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 

Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
'''
class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        l, sub = 0, ''
        for i in s:
            if i in sub:
                print(sub)
                sub = sub.split(i)[-1]
            sub += i
            if len(sub) > l:
                l = len(sub)
        return l

if __name__ == "__main__":
    solution = Solution()
    s = 'pwwkew'
    l = solution.lengthOfLongestSubstring(s)
    print(l)