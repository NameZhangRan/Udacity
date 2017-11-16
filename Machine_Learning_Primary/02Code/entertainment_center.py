#-*-coding:utf-8-*-
#设置支持中文,并使用Pycharm测试OK

import media
import fresh_tomatoes

#使用电影中文名的汉语拼音第一大写字母作为电影实例名字
#NO.1
SHGR = media.Movie("山河故人",
                   "贾樟柯导演的作品。讲述了发生在汾阳的一代人从1995年到2025年的情感和时代变化的故事。",
                   "http://bbs.e0514.com/data/attachment/forum/201510/29/151254lvwcqtqqoad5w9qo.jpg",
                   "https://youtu.be/pdavQLgSuRg")
print (SHGR.title)
print (SHGR.storyline)

#NO.2
DHXY = media.Movie("大话西游",
                   "周星驰的经典电影，用无厘头的方式阐述着默默情深。",
                   "http://shanxi.sinaimg.cn/2017/0208/U12851P1196DT20170208151135.jpg",
                   "https://youtu.be/XrReF3Eny8E")
print (DHXY.title)
print (DHXY.storyline)

#NO.3
HHWQ = media.Movie("后会无期",
                   "韩寒导演作品。有时候，你想证明给一万个人看，到后来，你发现只得到了一个明白的人，那就够了",
                   "http://i.ce.cn/ce/culture/gd/201407/26/W020140726277348711036.jpg",
                   "https://youtu.be/J7QHG12zjAg")
print (HHWQ.title)
print (HHWQ.storyline)

#NO.4
DYHT = media.Movie("大鱼海棠",
                   "该片讲述了掌管海棠花生长的少女椿为报恩而努力复活人类男孩“鲲”的灵魂，在本是天神的湫帮助下与彼此纠缠的命运斗争的故事。",
                   "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1495968832714&di=4af978b28f585f1db5d8ce563906b203&imgtype=0&src=http%3A%2F%2Fimg.zcool.cn%2Fcommunity%2F0123625789021b0000012e7ee1cc42.jpg",
                   "https://youtu.be/bH37VLQqVvo")
print (DYHT.title)
print (DYHT.storyline)

#NO.5
KFJ = media.Movie("空房间",
                  "韩国金基德导演作品。影片讲述了少妇善花和男孩泰石之间奇特的爱情故事。",
                  "https://imgsa.baidu.com/baike/c0%3Dbaike116%2C5%2C5%2C116%2C38/sign=68f2c531d688d43fe4a499a01c77b97e/43a7d933c895d14339ba866773f082025baf07a1.jpg",
                  "https://youtu.be/wIYMXpvC3uo")
print (KFJ.title)
print (KFJ.storyline)

#NO.6
CQSL = media.Movie("重庆森林",
                   "失恋的警察与神秘女杀手一段都市邂逅以及巡警与快餐店女孩的爱情故事。",
                   "https://imgsa.baidu.com/baike/c0%3Dbaike180%2C5%2C5%2C180%2C60/sign=a0955369ab6eddc432eabca958b2dd98/78310a55b319ebc419240b158426cffc1f1716d9.jpg",
                   "https://youtu.be/StZ9f6jQTR4")
print (CQSL.title)
print (CQSL.storyline)

#创建movies列表储存电影数据
movies = [SHGR, DHXY, HHWQ, DYHT, KFJ, CQSL]
#实现电影网站
fresh_tomatoes.open_movies_page(movies)


