# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/12 11:13 AM'
'''
ou are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order 
and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Example:
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
'''
# Python的链表
class Solution:
    def addTwoNumber(self,l1,l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1:
            return l2
        if not l2:
            return l1
        val1, val2 = [l1.val],  [l2.val]
        while l1.next:
            val1.append(l1.next.val)
            l1 = l1.next
        while l2.next:
            val2.append(l2.next.val)
            l2 = l2.next
        num1 = ''.join([str(i) for i in val1[::-1]])
        num2 = ''.join([str(i) for i in val2[::-1]])
        tmp = str(int(num1) + int(num2))[::-1]
        res = ListNode(int(tmp[0]))
        run_res = res
        for i in range(1,len(tmp)):
            run_res.next = ListNode(int(tmp[i]))
            run_res = run_res.next
        return res
