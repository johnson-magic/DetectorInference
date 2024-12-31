import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def add_angle_result(result: list) -> list:
    result = [x for x in result if x["confidence"] > 0.6]

    plates_res = [x for x in result if x["name"] == "plates"]
    assert len(plates_res) == 1
    plates_res = plates_res[0]  # 清爽一点
    
    slide_res = [x for x in result if x["name"] == "slide"]
    assert len(slide_res) == 1
    slide_res = slide_res[0]
    
    big_circle_res= [x for x in result if x["name"] == "big_circle"]
    assert len(big_circle_res) == 1
    big_circle_res = big_circle_res[0]
    
    
    plates_coors = [(plates_res["box"][f"x{i}"], plates_res["box"][f"y{i}"])
                       for i in range(1, 5)]
    plates_res["angle"] = calculate_rotation_angle_box(plates_coors)
    
    
    slide_coors = [(slide_res["box"][f"x{i}"], slide_res["box"][f"y{i}"])
                       for i in range(1, 5)]
    # slide_res["angle"] = calculate_rotation_angle_box(slide_coors) % 60
    # if slide_res["angle"] > 10:
    #     slide_res["angle"] = slide_res["angle"] - 60
    # elif slide_res["angle"] < -10:
    #     slide_res["angle"] = slide_res["angle"] + 60
    slide_res["angle"] = calculate_rotation_angle_box(slide_coors)
    
        
    result = []
    result.append(plates_res)
    result.append(slide_res)
    result.append(big_circle_res)
    
    return result

def calculate_box_center(vertices: list)-> tuple:
    """
    计算旋转矩形框的中心点坐标

    Args:
        vertices, list 四个顶点坐标的列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Return:
        中心点坐标 (cx, cy)
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]

    # 计算中心点坐标
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4

    return (cx, cy)

def calculate_line_center(vertices: list)-> tuple:
    """
    计算线段的中心点坐标

    Args:
        vertices, list 两个顶点坐标的列表，格式为 [(x1, y1), (x2, y2)]
    
    Return:
        中心点坐标 (cx, cy)
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]

    # 计算中心点坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return (cx, cy)

def calculate_distance(point1, point2):
    """
    计算两个点之间的距离.
    
    参数:
    point1: tuple，格式为 (x1, y1)
    point2: tuple，格式为 (x2, y2)
    
    返回:
    float，两个点之间的距离
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def get_res_infos(results: list) -> dict:
    infos = {}
    for res in results:
        if res["name"] == "big_circle":
            coors = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                       for i in range(1, 5)]
             
            center_point = calculate_box_center(coors)
            line_half_point = calculate_line_center(coors[:2])
            radius = calculate_distance(center_point, line_half_point)
            
            infos["CenterPoint"] = center_point
            infos["Diameter"] = 2 * radius
        elif res["name"] == "plates":
            infos["Angle"] = res["angle"]
        elif res["name"] == "slide":
            infos["SliderAngle"] = res["angle"]
            coors = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                       for i in range(1, 5)]
            infos["SliderCenterPoint"] = calculate_box_center(coors)
    
    if infos["SliderCenterPoint"][1] > infos["CenterPoint"][1]:
        infos["Position"] = "below"
    elif infos["SliderCenterPoint"][1] <= infos["CenterPoint"][1]:
        infos["Position"] = "above"
    
    return infos


import hashlib

def calculate_md5(image_path):
    """
    计算图像文件的MD5哈希值.
    
    参数:
    image_path: str，图像文件的路径
    
    返回:
    str，图像文件的MD5哈希值
    """
    # 创建一个md5哈希对象
    md5_hash = hashlib.md5()
    
    # 分块读取文件，更加高效
    with open(image_path, "rb") as image_file:
        # 每次读取4096字节
        for byte_block in iter(lambda: image_file.read(4096), b""):
            md5_hash.update(byte_block)
    
    # 返回MD5哈希值的十六进制表示
    return md5_hash.hexdigest()

def save_res_infos(res_infos, save_path):
    with open(save_path, 'w') as file:
        file.write(f"CenterPoint {res_infos['CenterPoint'][0]},{res_infos['CenterPoint'][1]}\n")
        file.write(f"Angle {res_infos['Angle']}\n")
        file.write(f"Position {res_infos['Position']}\n")
        file.write(f"Diameter {res_infos['Diameter']}\n")
        file.write(f"SliderAngle {res_infos['SliderAngle']}\n")


import math

def convert_pixel_to_cartesian(coors: list) -> list:
    """图片上的坐标，y方向的大小与笛卡尔坐标系是反着的，该函数用于调整这种相对大小关系。
    
    Args:
        coors: list, 像素坐标系中的坐标[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Return:
        list, （仅逻辑上）笛卡尔坐标系的坐标
    """
    converted_coors = [(x, -y) for (x, y) in coors]
    return converted_coors
    

def calculate_rotation_angle_box(coors: list) -> float:
    """通过box的四个顶点，计算box的旋转角度(逆时针为正)
    
    Args:
        coors: list, box的四个顶点的坐标（像素坐标系）[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Return:
        float, box的旋转角度
    """
    assert len(coors) == 4
    
    coors = convert_pixel_to_cartesian(coors)
    x1, y1, x2, y2, x3, y3 = coors[0][0], coors[0][1], coors[1][0], coors[1][1], coors[2][0], coors[2][1]
   
    edg1_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    edg2_len = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    
    start_x = x1
    start_y = y1
    end_x = x2
    end_y = y2
    
    if edg1_len < edg2_len:
        start_x = x2
        start_y = y2
        end_x = x3
        end_y = y3    
    
    
    angle_deg = calculate_rotation_angle(start_x, start_y, end_x, end_y)

    if angle_deg % 90 == 0:
        angle_deg = 0

    return angle_deg

def calculate_rotation_angle_line(coors):
    """通过line的两个顶点，计算line的旋转角度(逆时针为正)
    
    Args:
        coors: list, line的两个顶点的坐标（像素坐标系）[(x1, y1), (x2, y2)]
    
    Return:
        float, line的旋转角度
    """
    assert len(coors) == 2
    coors = convert_pixel_to_cartesian(coors)
    
    start_x, start_y, end_x, end_y = coors[0][0], coors[0][1], coors[1][0], coors[1][1]
    
    angle_deg = calculate_rotation_angle(start_x, start_y, end_x, end_y)
    
    if angle_deg % 90 == 0:
        angle_deg = 0

    return angle_deg
    
    

def calculate_rotation_angle(start_x, start_y, end_x, end_y):
    # 计算两个点之间的向量
    dx = end_x - start_x
    dy = end_y - start_y

    angle_rad = math.atan(dy/dx)

    # 将弧度转换为角度
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_center(vertices: list)-> tuple:
    """
    计算旋转矩形框的中心点坐标

    Args:
        vertices, list 四个顶点坐标的列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Return:
        中心点坐标 (cx, cy)
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]

    # 计算中心点坐标
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4

    return (cx, cy)

def draw_rotated_rectangle(image_path, vertices, color, save_path):
    """
    在图像上绘制旋转矩形框

    :param image_path: 图片的路径
    :param vertices: 四个顶点坐标的列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param color: 绘制矩形框的颜色，格式为 (r, g, b)
    """
    # 打开图像
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # 绘制矩形框
        draw.line(vertices + [vertices[0]], fill=color, width=2)  # 连接最后一个点到第一个点

        # 显示或保存结果图像
        # img.show()  # 显示图片
        img.save(save_path)  # 可以选择保存图片
        #return img

def draw_label(image_path: str, vertices: list, label: str, color: tuple, save_path: str)-> None:
    """在第一个点上渲染label信息
    
    image_path: str, （待绘制）图片路径
    vertices: list,  [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    label: 待渲染label信息
    color: 渲染文字颜色
    save_path: str, （渲染）图片保存路径
    """
    first_point = vertices[0]
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        font_path = "Arial.Unicode.ttf"
        size = max(round(sum(img.size) / 2 * 0.035), 12)
        font = ImageFont.truetype(str(font_path), size)
    
        w, h = font.getsize(label)  # text width, height
        outside = first_point[1] >= h  # label fits outside box
        if first_point[0] > img.size[0] - w:  # size is (w, h), check if label extend beyond right side of image
            first_point = (img.size[0] - w, first_point[1])
        draw.rectangle((first_point[0], first_point[1] - h if outside else first_point[1],
                        first_point[0] + w + 1, first_point[1] + 1 if outside else first_point[1] + h + 1),
                fill=color,
            )
            
        draw.text((first_point[0], first_point[1] - h if outside else first_point[1]), label, fill=(0, 0, 0), font=font)
    img.save(save_path)

    
def draw_axes(image_path, center, save_path):
    """
    在图像上以给定点为中心绘制坐标轴

    :param image_path: 图片的路径
    :param center: 中心点坐标 (x, y)
    """
    # 打开图像
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # 中心点坐标
        x, y = center
        
        # 坐标轴长度（15个像素，纵横各一半，总共30个像素）
        length = 15  

        # 绘制X轴（横轴）
        draw.line([(x - length, y), (x + length, y)], fill="black", width=1)

        # 绘制Y轴（纵轴）
        draw.line([(x, y - length), (x, y + length)], fill="black", width=1)

        # 显示结果图像
        # img.show()  # 显示图片
        # img.save('output_image.jpg')  # 可以选择保存图片
        img.save(save_path)
    
    

def vis_res_infos(json_result, src_path, vis_path):
    # Open the BMP image
    with Image.open(src_path) as img:
        # Convert to RGB (JPEG does not support transparency)
        rgb_image = img.convert('RGB')
        # Save as JPG
        rgb_image.save(vis_path, 'JPEG')
    for res in json_result:
        if res["name"] == "plates":
            # 绘制预测框，和角度
            angle = res["angle"]
            coor = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                    for i in range(1, 5)]
            draw_rotated_rectangle(vis_path, coor, (11, 219, 235), vis_path)
            label = res["name"] + " " + str(round(res["confidence"], 2)) + " " + str(round(angle, 2)) 
            draw_label(vis_path, coor, label, (11, 219, 235), vis_path)
            draw_axes(vis_path, coor[2], vis_path)
                    
        elif res["name"] == "slide":
                # 绘制预测框，和角度
            angle = res["angle"]
            coor = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                    for i in range(1, 5)]
            draw_rotated_rectangle(vis_path, coor, (243, 243, 243), vis_path)
            label = res["name"] + " " + str(round(res["confidence"], 2)) + " " + str(round(angle, 2)) 
            draw_label(vis_path, coor, label, (243, 243, 243), vis_path)
            draw_axes(vis_path, coor[2], vis_path)
            
        elif res["name"] == "big_circle":
            # 绘制预测框，和角度
            # angle = res["angle"]
            coor = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                    for i in range(1, 5)]
            draw_rotated_rectangle(vis_path, coor, (4, 42, 255), vis_path)
            label = res["name"] + " " + str(round(res["confidence"], 2)) # + " " + str(round(angle, 2)) 
            draw_label(vis_path, coor, label, (4, 42, 255), vis_path)
            draw_axes(vis_path, coor[2], vis_path)

def write_json(res, res_path):
    with open(res_path, "w") as fw:
	    json.dump(res, fw, ensure_ascii = False, indent = 4)
    
    
    
            
            
            
            
            
    