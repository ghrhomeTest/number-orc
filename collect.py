import json
import time


p = r'tests\mytest\data.txt'
p2 = r'tests\mytest\data.json'
d = '1233'


def write_data(path, data):
    f = open(path, 'a')
    f.write(data+'\n')
    f.close()


# 图片名使用时间命名
# 这段测试需要删掉

# nowtime = time.strftime('%d-%H-%M-%S', time.localtime())
# p2 = r'tests\mytest\data.json'
# test_dict = {nowtime: 990}
# fr = open(p2)
# model = json.load(fr)
# fr.close()
#
# with open(p2, 'w') as f:
#     json.dump(test_dict, f)
#     print('yes')


def write_json(path2, data1, data2):
    obj = {data1: data2}
    fr = open(path2)
    model = json.load(fr)
    fr.close()
    for i in obj:
        model[i] = obj[i]
    jsobj = json.dumps(model)
    with open(path2, 'w') as fw:
        fw.write(jsobj)
        fw.close()
