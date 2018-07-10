# _*_ coding=utf-8 _*_
import os
import sys
import codecs
from bs4 import BeautifulSoup
reload(sys)
sys.setdefaultencoding('utf-8')

path = 'D:/Users/sangfor/Desktop/TextInfoExp-master/TextInfoExp-master/Part1_TF-IDF/data/computer/'
path1 = 'D:/Users/sangfor/Desktop/TextInfoExp-master/TextInfoExp-master/Part1_TF-IDF/data/title_and_abs/'
newpath = 'D:/Users/sangfor/Desktop/TextInfoExp-master/TextInfoExp-master/Part1_TF-IDF/data/pro_keyword/'
filelist = os.listdir(path)
# 清洗出xml格式的文件中的标题和摘要信息
def get_text():
    abstracts = []
    for files in filelist:
        filename = os.path.splitext(files)[0]  # 取文件名
        soup = BeautifulSoup(open(path + filename + '.xml'), 'html.parser')  # 解析网页
        b = soup.find("p", class_="abstracts")  # 取得"p", class_="abstracts"为标签的内容
        # print b
        if b is None or b.string is None:
            continue
        else:
            abstracts.extend(soup.title.stripped_strings)
            s = b.string
            abstracts.extend(s)
            f = codecs.open(path1 + filename + ".txt", "w+", 'utf-8')  # 写入txt文件,使用codecs的方法进行解析，可以避免出现文件中ascii类型的值无法读入的情况
            listtext = []
            for i in abstracts:
                listtext.append(i)
                f.write(i)
            f.close()
         
            abstracts = []

        # getPro_keyword，清洗出xml文件中dl标签中的文本信息
        links = soup.find_all("dl")
        # print links
        for link in links:
            s1 = link.get_text()
            # print s1
        f = codecs.open(newpath + filename + ".txt", "w+",'utf-8')  # 将得到的未处理的文字放在pro_keyword文件夹中
        for i in s1:
            f.write(i)
        f.close()
get_text()
