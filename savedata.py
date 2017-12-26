#encoding=utf-8
#example for datasave
import pymssql

conn=pymssql.connect(host='.',user='sa',password='123456',database='ReportServerTempDB')
cur = conn.cursor()

#change the sql string to save data
cur.execute('select top 5 * from [dbo].[ChunkData]')
print(cur.fetchall())
cur.close()
conn.close()