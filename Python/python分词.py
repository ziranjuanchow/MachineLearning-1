# _*_ encoding:utf-8 _*_
import jieba_fast as jieba 
import jieba.analyse
import json
# 读取文本数据，使用字典保存词频，并将字典中的数据按行写入txt文本

filename = u'/saasdata/**/data/file_title.txt'

word_dict = {}

with open(filename,'r') as f:
	for item in jieba.cut(f.read()):
		num = word_dict.get(item.encode('utf-8'), 0)
		word_dict[item] = num+1


	tmp = sorted(word_dict.items(), key= lambda a: a[1], reverse=True)
	word_dict = dict(tmp)
fileobject = open(u'/saasdata/**/data/count_word.txt', 'w')
for key,values in word_dict.items():
	print(key,values)
	fileobject.writelines(str(key.encode('utf-8')) + ' '+str(values)+ '\n')
# json_data =  json.dumps(word_dict)
# json_data.encode('utf-8')
print('Sucessful saved!!!')
	
