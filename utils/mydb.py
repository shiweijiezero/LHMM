import pymongo
import pytz
from config import DefaultConfig

class MYDB():
    """
    MongoDB数据库
    """
    def create_instance(self,dbname):
        return pymongo.MongoClient(self.DBURL, tz_aware=True,serverSelectionTimeoutMS=60000, socketTimeoutMS=60000,
                                   tzinfo=pytz.timezone('Asia/Shanghai'))[
            dbname]

    def __init__(self, DBURL=DefaultConfig.dbsession_path,dbname = "spark"):
        self.DBURL = DBURL
        self.db = self.create_instance(dbname)




