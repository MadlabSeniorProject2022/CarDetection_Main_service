import cv2
import time
import datetime

import os
from pathlib import Path

import concurrent
import requests

from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./carrgclassification-857e3c375cdd.json'

import json

import database as db
from lean_detect import UseModel

VEHICLE_DETECTION_MODEL = UseModel("yolov7.pt", detect_class=[2, 5, 7], confident=0.75)

# Opening JSON file
with open('car_meta.json') as json_file:
    car_meta = json.load(json_file)

def get_meta(img: cv2.Mat):
    image_data = cv2.imencode('.jpg', img)[1].tobytes()
    urls = [
        
        "https://72c4-34-143-252-179.ngrok.io/useColor",
        "https://52c7-35-247-181-246.ngrok.io/predict"
    ]

    def request_post(url, data):
        return requests.post(url, files={"image": data}).json()

    with concurrent.futures.ThreadPoolExecutor() as executor: # optimally defined number of threads
        res = [executor.submit(request_post, url, image_data) for url in urls]
        concurrent.futures.wait(res)

    return {"color": res[0].result()["predicted"], "model": res[1].result()["predicted"]}

def upload_cs_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)

def get_cs_file_url(bucket_name, file_name, expire_in=datetime.datetime.now() + datetime.timedelta(4000)): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    url = bucket.blob(file_name).generate_signed_url(expire_in)

    return url

def cloud_image (bucket_name, source_file_name, destination_file_name):
    upload_cs_file(bucket_name, source_file_name, destination_file_name)
    url = get_cs_file_url(bucket_name, destination_file_name)
    return url



def car_works (coor, img, conf, cls):
    current_time = int(round(time.time() * 1000))
    print(current_time)
    if not os.path.exists("./runs/"):
        os.makedirs("./runs/")
    Path("./runs/realuse/").mkdir(parents=True, exist_ok=True)
    crop_img = img[ int(coor[1]) : int(coor[3]), int(coor[0]): int(coor[2])]
    #result = cv2.resize(crop_img, (640, 640))
    #result = cv2.cvtColor(result , cv2.COLOR_RGB2GRAY)
    meta = get_meta(crop_img)
    print(meta)
    car_detail = car_meta[meta["model"]]
    print(car_detail)
    cv2.imwrite(f"./runs/realuse/{current_time}.jpg", img)
    origin_img_path = cloud_image('images-bucks', f"./runs/realuse/{current_time}.jpg", f'{current_time}-origin.jpg')
    cv2.imwrite(f"./runs/realuse/{current_time}-crop.jpg", crop_img)
    crop_img_path = cloud_image('images-bucks', f"./runs/realuse/{current_time}-crop.jpg", f'{current_time}-crop.jpg')

    db.update_data(car_detail["make"], car_detail["model"], car_detail["class"], "Unknown", meta["color"], origin_img_path, None, crop_img_path)
    

def main_detect_car_step (source):
    VEHICLE_DETECTION_MODEL.detect(source, do_function=car_works)
