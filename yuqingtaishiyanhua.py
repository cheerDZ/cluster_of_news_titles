import pandas as pd 
import numpy as np 
import time 
from IPython.display import display

#数据处理
# df=pd.read_csv('/Users/apple/Desktop/短文本聚类/民航新闻.csv')
# display(df.head())
# display(df.info())
# df=df[['issueTime','newsTitle']]
# display(df.head())
# df=df.drop_duplicates('newsTitle',keep='first',inplace=False)
# news=pd.Series(data=df['newsTitle'].values)
# i=0
# file=open('minhang_news.txt','a') 
# while i<914:
#     new=news[i]
#     i=i+1
#     file.write(new+'\n')
# file.close()
startTime=time.time()
#创建停用词
stopwords_filepath='/Users/apple/Desktop/短文本聚类/stopwordslist.txt'
def stopwordslist(stopwords_filepath):
    stopwords=[line.strip() for line in open(stopwords_filepath,'r',encoding='utf-8').readlines()]
    return stopwords

#对句子进行分词
userdict_filepath='/Users/apple/Desktop/短文本聚类/userdict.txt'
def segment(text,userdict_filepath,stopwords_filepath):
    import jieba
    jieba.load_userdict(userdict_filepath)
    stopwords=stopwordslist(stopwords_filepath)
    seg_list=jieba.cut(text,cut_all=False)
    seg_list_without_stopwords=[]
    for word in seg_list:
        if word not in stopwords:
            if word !='\t':
                seg_list_without_stopwords.append(word)
    return seg_list_without_stopwords

#使用分词器将files进行分词
f=open('minhang_news.txt','r',encoding='utf-8')
files=f.readlines()
files=pd.Series(files)
#先按想要的字段聚焦于更小的一类
bool=files.str.contains('北京')
files=files[bool]
print(files)
files=files.tolist()

totalvocab_tokenized=[]
for i in files:
    allwords_tokenized=segment(i,userdict_filepath,stopwords_filepath)
    totalvocab_tokenized.extend(allwords_tokenized)
#显示分词结果
display(totalvocab_tokenized)
#显示分词后的词语数量
display(len(totalvocab_tokenized))
#显示分词花费的时间
print('分词花费的时间为：%.2f秒' % (time.time()- startTime))
#获得TF_IDF矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords_list=[k.strip() for k in open(stopwords_filepath,encoding='utf-8').readlines() if k.strip() !='']
tfidf_vectorizer=TfidfVectorizer(totalvocab_tokenized,
                                stop_words=stopwords_list,
                                min_df=0,
                                max_df=0.9,
                                max_features=200000,
                                )

tfidf_matrix = tfidf_vectorizer.fit_transform(files)
print(tfidf_matrix.shape)

#计算文档相似性
from sklearn.metrics.pairwise import cosine_similarity
#两个向量越相似，夹角越小，余弦值越大，dist越小
dist=1-cosine_similarity(tfidf_matrix)

#获得分类
from scipy.cluster.hierarchy import ward,dendrogram,linkage
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['SimHei']
def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
#采用ward(最短最长平均离差平方和)
linkage_matrix=linkage(dist,method='single',metric='euclidean',optimal_ordering=False)
print(linkage_matrix)

#可视化
plt.figure()
plt.title('新闻标题聚类树状图',fontproperties=getChineseFont())
plt.xlabel('新闻标题',fontproperties=getChineseFont())
plt.ylabel('距离（越低表示文本越类似）',fontproperties=getChineseFont())
dendrogram(
    linkage_matrix,
    labels=files,
    leaf_rotation=70,
    leaf_font_size=12
)
plt.show()
plt.close()