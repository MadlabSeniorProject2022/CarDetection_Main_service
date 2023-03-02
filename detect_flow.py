import cv2
import time
import os
from pathlib import Path

import concurrent
import requests
import datetime

import database as db

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./carrgclassification-857e3c375cdd.json'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from lean_detect import UseModel

# Loading Vehicle Model
vehicle_detect_model = UseModel("yolov7.pt", detect_class=[2, 5, 7], confident=0.75)

# Loading Car models meta data
import json
 
# Opening JSON file
with open('car_meta.json') as json_file:
    car_meta = json.load(json_file)

def get_meta(img: cv2.Mat):
    image_data = cv2.imencode('.jpg', img)[1].tobytes()
    urls = [
        
        "https://72c4-34-143-252-179.ngrok.io/useColor", #Color
        "https://52c7-35-247-181-246.ngrok.io/predict", #Model
        "https://8fcb-35-193-242-237.ngrok.io/predict" #Plate

    ]

    def request_post(url, data):
        return requests.post(url, files={"image": data}).json()

    with concurrent.futures.ThreadPoolExecutor() as executor: # optimally defined number of threads
        res = [executor.submit(request_post, url, image_data) for url in urls]
        concurrent.futures.wait(res)

    return {
        "color": res[0].result()["predicted"], 
        "model": res[1].result()["predicted"],
        "possible": res[1].result()["possible"], 
        "plate_num": res[2].result()["plate_num"],
        "plate_url": res[2].result()["plate_url"] }


# Push Image to Google Cloud Storage
def upload_cs_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)

# Generate Image URL from Google Cloud Storage
def get_cs_file_url(bucket_name, file_name, expire_in=datetime.datetime.now() + datetime.timedelta(4000)): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    url = bucket.blob(file_name).generate_signed_url(expire_in)

    return url

# Mix them all for Google Cloud Storage's flow
def cloud_image (bucket_name, source_file_name, destination_file_name):
    upload_cs_file(bucket_name, source_file_name, destination_file_name)
    url = get_cs_file_url(bucket_name, destination_file_name)
    return url

def detection_tasks (coor, img, conf, cls):
    current_time = int(round(time.time() * 1000))
    print(current_time)
    if not os.path.exists("./runs/"):
        os.makedirs("./runs/")
    Path("./runs/exp/").mkdir(parents=True, exist_ok=True)
    crop_img = img[ int(coor[1]) : int(coor[3]), int(coor[0]): int(coor[2])]
    #result = cv2.resize(crop_img, (640, 640))
    #result = cv2.cvtColor(result , cv2.COLOR_RGB2GRAY)
    meta = get_meta(crop_img)
    print(meta)
    car_detail = car_meta[meta["model"]]
    possible = [{**car_meta[pos["class"]], "conf" : pos["conf"]} for pos in meta["possible"]]
    print(car_detail)
    cv2.imwrite(f"./runs/exp/{current_time}.jpg", img)
    origin_img_path = cloud_image('images-bucks', f"./runs/exp/{current_time}.jpg", f'{current_time}-origin.jpg')
    cv2.imwrite(f"./runs/exp/{current_time}-crop.jpg", crop_img)
    crop_img_path = cloud_image('images-bucks', f"./runs/exp/{current_time}-crop.jpg", f'{current_time}-crop.jpg')

    db.update_data(car_detail["make"], car_detail["model"], car_detail["class"], meta["plate_num"], meta["color"], origin_img_path, meta["plate_url"], crop_img_path, current_time, possible)

def detect_flow (source):
    vehicle_detect_model.detect(source, do_function=detection_tasks)
    