import pymongo
from bson.objectid import ObjectId

myclient = pymongo.MongoClient("mongodb+srv://thanick_ku:kukuru2000@cluster0.yevnu7y.mongodb.net/?retryWrites=true&w=majority")

mydb = myclient["CDT"]
mycol = mydb["CDT"]

def get_list_data ():
    return mycol.find()

def get_lasted ():
    return mycol.find_one({}, sort=[( '_id', pymongo.DESCENDING )])

def get_byid (id):
    return mycol.find_one(ObjectId(id))

def update_data (make: str, model: str, type: str, license_num: str, color: str, path: str, lp_path: str, car_path: str):
    new_data = {
        "make": make,
        "model": model,
        "class": type,
        "license_num": license_num,
        "color": color,
        "path": path,
        "lp_path": lp_path,
        "car_path": car_path
    }
    x = mycol.insert_one(new_data)
    print(x)