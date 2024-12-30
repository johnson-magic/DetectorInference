import os
import json
import time
from ultralytics import YOLO
from utils.utils import add_angle_result, calculate_md5, get_res_infos, save_res_infos, vis_res_infos

model_path = "./best-cpu.onnx"
img_path = "./imgs/test.bmp"
save_path = "./res/data.txt"
vis_path = "./res/vis.jpg"
speed_test = True

model = YOLO(model_path, task='obb')

old_md5_code = -1

try:
    print("开始循环，按 Ctrl + C 终止程序...")
    while True:
        if not os.path.exists(img_path):
            continue
        new_md5_code = calculate_md5(img_path)
        
        # 测试耗时
        run_iter = 5 if speed_test else 1
        
        if old_md5_code != new_md5_code:
            old_md5_code = new_md5_code
            print("检测到新图片...")
            
            if speed_test:
                start = time.time()
            for i in range(run_iter):
                results = model(img_path) # predict on an image
            if speed_test:
                end_inference = time.time()
            
            for i in range(run_iter):
                json_result = results[0].to_json()  # str
                json_result = json.loads(json_result)  # json obj
                    
                    
                json_result = add_angle_result(json_result)  # 增添了angle
                res_infos = get_res_infos(json_result)
            if speed_test:
                end_process = time.time()
            
            for i in range(run_iter):
                save_res_infos(res_infos, save_path)
            if speed_test:
                end_saveres = time.time()
            
            for i in range(run_iter):
                vis_res_infos(json_result, img_path, vis_path)
            if speed_test:
                end_visres = time.time()
            
            if speed_test:
                print(f"total timecost: {(end_visres - start) / run_iter * 1000} ms")
                print(f"all of inferencer timecost: {(end_inference - start) / run_iter * 1000} ms")
                print(f"process of angle detector timecost: {(end_process - end_inference) / run_iter * 1000} ms")
                print(f"save result in txt of angle detector timecost: {(end_saveres - end_process) / run_iter * 1000} ms")
                print(f"save result in image of angle detector timecost: {(end_visres - end_saveres) / run_iter * 1000} ms")
            
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n循环已终止。")
    
    
