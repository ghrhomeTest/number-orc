from playground import main
import pymysql
from collect import write_data, p, p2, write_json


# file_name = 'tests/mytest/t7.jpg'
def thismain(file_name,nowtime):
    re = main(file_name)
    write_data(p, re)
    write_json(p2, nowtime, re)
    print(re)


if __name__ == '__main__':
    thismain()
