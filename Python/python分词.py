# _*_ encoding:utf-8 _*_
import jieba_fast as jieba 
import jieba.analyse
import json
'''
读取文本文件，使用jieba进行分词。保存为dict字典，使用json来对结果保存。
技巧：
    使用json来处理dict的数据，处理很方便，直接可以保存为文本类文件
'''

filename = '/saasdata/liudong/data/file_title.txt'

word_dict = {}

with open(filename,'r') as f:
	for item in jieba.cut(f.read()):
		num = word_dict.get(item, 0)
		word_dict[item] = num+1


	tmp = sorted(word_dict.items(), key= lambda a: a[1], reverse=True)
	word_dict = dict(tmp)
fileobject = open('/saasdata/liudong/data/count_word.txt', 'w')
fileobject.writelines(json.dumps(word_dict)+'\n')
print 'Sucessful saved!!!'
