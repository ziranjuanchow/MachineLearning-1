# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2019/1/6 10:50 PM'

import numpy as np

class HiddenMarkov:
    def forward(self,Q,V,A,B,O,PI):
        """
        使用前向算法
        :param Q: 状态序列
        :param V: 观测序列
        :param A: 
        :param B: 
        :param O: 
        :param PI: 
        :return: 
        """
        N = len(Q)
        M = len(O)
        alphas = np.zeros(N, M)
        T = M # 观测时刻
        # 遍历每一时刻，算出alpha值
        for t in range(T):
            indexOfO = V.index(O[t])
            for i in range(N):
                # 计算初值
                if t == 0:
                    alphas[i][t] = PI[t][i] * B[i][indexOfO]
                    print('alpha1(%d)=p%db%db(o1)=%f'%(i,i,i,alphas[i][t]))
                else:
                    alphas[i][t] = np.dot([])

