# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/10/13 4:28 PM'
'''
Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
'''

def reverseList(self,head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    cur, prev = head, None
    while cur:
        cur.next, prev, cur = prev, cur, cur.next
    return prev
