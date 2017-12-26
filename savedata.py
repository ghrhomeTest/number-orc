#encoding=utf-8
#example for datasave
#最后需要写一个datasave的services 在开机之后就运行
#注意判断各种状态 以及不能存储的数据
#表的结构 需要设定好 主键为设备id 应从usb摄像头id得到对应关系
import pymssql

conn=pymssql.connect(host='.',user='sa',password='123456',database='ReportServerTempDB')
cur = conn.cursor()

#change the sql string to save data
cur.execute('select top 5 * from [dbo].[ChunkData]')
print(cur.fetchall())
cur.close()
conn.close()