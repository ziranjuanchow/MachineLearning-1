# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/16 10:36 AM'
'''
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
'''
import re
'''
使用re.match(prefix，strs[i]) 
对prefix,strs[i]进行匹配
 当不匹配的时候，减少prefix的长度进行继续匹配
 最后输出减少后的prefix的值
'''
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ''
        prefix = strs[0]
        for i in range(1, len(strs)):
            match = re.match(prefix,strs[i])
            while not match:
                prefix = prefix[:-1]
                match = re.match(prefix,strs[i])
        return prefix

if __name__ == "__main__":
    solution = Solution()
    strs = ["flower","flow","flight"]
    result = solution.longestCommonPrefix(strs)
    print(result)