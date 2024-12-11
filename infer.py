import os
import json
import time
from ultralytics import YOLO
from utils.utils import add_angle_result, calculate_md5, get_res_infos, save_res_infos, vis_res_infos

model_path = "./best-cpu.onnx"
img_path = "./imgs/test.bmp"
save_path = "./res/data.txt"
vis_path = "./res/vis.jpg"
model = YOLO(model_path, task='obb')

old_md5_code = -1

try:
    print("开始循环，按 Ctrl + C 终止程序...")
    while True:
        if not os.path.exists(img_path):
            continue
        new_md5_code = calculate_md5(img_path)
        
        if old_md5_code != new_md5_code:
            old_md5_code = new_md5_code
            print("检测到新图片...")
            
            results = model(img_path) # predict on an image
            
            json_result = results[0].to_json()  # str
            json_result = json.loads(json_result)  # json obj
                
                
            json_result = add_angle_result(json_result)  # 增添了angle
            res_infos = get_res_infos(json_result)
            
            save_res_infos(res_infos, save_path)
            
            vis_res_infos(json_result, img_path, vis_path)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n循环已终止。")
    
    
