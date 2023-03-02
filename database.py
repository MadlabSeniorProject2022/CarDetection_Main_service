import pymongo
from bson.objectid import ObjectId

myclient = pymongo.MongoClient("mongodb+srv://thanick_ku:kukuru2000@cluster0.yevnu7y.mongodb.net/?retryWrites=true&w=majority")

mydb = myclient["CDT"]
mycol = mydb["CDT"]

def get_list_data (query = {}):
    return mycol.find(query, sort=[( '_id', pymongo.DESCENDING )])

def get_lasted (query = {}):
    return mycol.find_one(query, sort=[( '_id', pymongo.DESCENDING )])

def get_byid (id):
    return mycol.find_one(ObjectId(id))

def update_human_edit (id, make, model, color, plate):
    myquery = {"_id" : ObjectId(id)}
    newvalues = { "$set": { 
        "real_make": make,
        "real_model": model,
        "real_color": color,
        "real_license_num": plate,
        "status": "success"
        } }
    mycol.update_one(myquery, newvalues)

def set_bad_cond (id):
    myquery = {"_id" : ObjectId(id)}
    newvalues = { "$set": { 
        "status": "bad_cond"
        } }
    mycol.update_one(myquery, newvalues)

def update_data (make: str, model: str, type: str, license_num: str, color: str, path: str, lp_path: str, car_path: str, time: int, possible: list):
    new_data = {
        "make": make,
        "model": model,
        "class": type,
        "possible": possible,
        "status": 'pending',
        "license_num": license_num,
        "color": color,
        "path": path,
        "lp_path": lp_path,
        "car_path": car_path,
        "time": time
    }
    x = mycol.insert_one(new_data)
    print(x)