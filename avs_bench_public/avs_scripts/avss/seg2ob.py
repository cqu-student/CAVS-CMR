import numpy as np
from PIL import Image
import pdb
import cv2
def closing_operation(image, kernel_size=(20, 20)):
    """
    对图像进行闭操作
    :param image: 输入的灰度图像 (NumPy 数组)
    :param kernel_size: 结构元素的尺寸，默认为 (5, 5)
    :return: 经过闭操作处理后的图像
    """
    # 定义结构元素 (矩形)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 闭操作：先膨胀后腐蚀
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return closed_image
image_path = "../../avsbench_data/2.png"
image_path2 = "../../avsbench_data/5.png"
gray_image1 = Image.open(image_path).convert("L")
gray_image2 = Image.open(image_path2).convert("L")
segmentation_label1 = np.array(gray_image1)
segmentation_label2 = np.array(gray_image2)
segmentation_label1 = closing_operation(segmentation_label1)
segmentation_label2 = closing_operation(segmentation_label2)

def get_bounding_box(binary_mask):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    # 找到目标区域的最小和最大行、列索引
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin, rmax, cmin, cmax
def find_connected_components(binary_mask):
    # 使用连通区域分析
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # 提取每个连通区域的掩码
    regions = []
    for i in range(1, num_labels):  # 跳过背景（label=0）
        region_mask = (labels == i).astype(np.uint8)
        # 计算当前区域的宽度和高度
        regions.append(region_mask)
    return regions

def segment_to_bouding(segmentation_label):
    unique_classes = np.unique(segmentation_label)
    unique_classes = unique_classes[unique_classes != 0]
    class_masks = {}
    for cls in unique_classes:
        class_masks[cls] = (segmentation_label == cls).astype(np.uint8)

    bounding_boxes = {}
    for cls, mask in class_masks.items():
        regions = find_connected_components(mask)
        bounding_boxes[cls] = []
        for region in regions:
            if np.any(region):  # 如果区域中存在非零像素
                rmin, rmax, cmin, cmax = get_bounding_box(region)
                bbox = [cmin, rmin, cmax, rmax]
                bounding_boxes[cls].append(bbox)
        # 输出每个类别的边界框
        for cls, bboxes in bounding_boxes.items():
            print(f"Class {cls} Bounding Boxes:")
            for bbox in bboxes:
                print(bbox)
    return bounding_boxes

def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的交并比 (IoU)
    :param bbox1: 第一个边界框 [x_min, y_min, x_max, y_max]
    :param bbox2: 第二个边界框 [x_min, y_min, x_max, y_max]
    :return: IoU 值
    """
    # 获取两个边界框的坐标
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 计算交集的左上角和右下角坐标
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 计算交集的宽度和高度
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)

    # 交集面积
    inter_area = inter_width * inter_height

    # 计算两个边界框的面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 并集面积
    union_area = area1 + area2 - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou
def find_high_overlap_boxes(bboxes1, bboxes2, threshold=0.8):
    """
    查找两个边界框字典中重合度大于阈值的边界框对，并保存类别信息
    :param bboxes1: 第一个图片的边界框字典 {class: [bbox1, bbox2, ...]}
    :param bboxes2: 第二个图片的边界框字典 {class: [bbox1, bbox2, ...]}
    :param threshold: 重合度阈值 (IoU 阈值)
    :return: 重合度大于阈值的边界框对列表 [(bbox1, class1, bbox2, class2, iou), ...]
    """
    high_overlap_pairs = []

    # 遍历第一个图片的边界框字典
    for cls1, boxes1 in bboxes1.items():
        for bbox1 in boxes1:
            # 遍历第二个图片的边界框字典
            for cls2, boxes2 in bboxes2.items():
                for bbox2 in boxes2:
                    # 计算 IoU
                    iou = calculate_iou(bbox1, bbox2)
                    if iou > threshold:
                        high_overlap_pairs.append((bbox1, cls1, bbox2, cls2, iou))

    return high_overlap_pairs
def non_max_suppression_with_area(bboxes, iou_threshold=0.5):
    """
    非极大值抑制 (NMS)，保留面积最大的框
    :param bboxes: 边界框列表，格式为 [(bbox, area), ...]
    :param iou_threshold: IoU 阈值，默认为 0.5
    :return: 经过 NMS 处理后的边界框列表
    """
    if len(bboxes) == 0:
        return []

    # 按照面积降序排序
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # 初始化保留的边界框列表
    keep_bboxes = []

    # 遍历边界框
    while len(bboxes) > 0:
        # 取出当前面积最大的边界框
        current_bbox, current_area = bboxes.pop(0)
        keep_bboxes.append((current_bbox, current_area))

        # 计算当前边界框与其他边界框的 IoU
        filtered_bboxes = []
        for bbox, area in bboxes:
            iou = calculate_iou(current_bbox, bbox)
            if iou < iou_threshold:
                filtered_bboxes.append((bbox, area))

        # 更新边界框列表
        bboxes = filtered_bboxes

    return keep_bboxes

def apply_nms_to_bounding_boxes(bounding_boxes, iou_threshold=0.1):
    """
    对每个类别的边界框应用 NMS
    :param bounding_boxes: 边界框字典，格式为 {class: [[x_min, y_min, x_max, y_max], ...]}
    :param iou_threshold: NMS 的 IoU 阈值，默认为 0.5
    :return: 经过 NMS 处理后的边界框字典
    """
    filtered_bounding_boxes = {}

    for cls, bboxes in bounding_boxes.items():
        # 计算每个边界框的面积
        bboxes_with_areas = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            area = (x_max - x_min) * (y_max - y_min)  # 计算面积
            bboxes_with_areas.append((bbox, area))

        # 对边界框应用 NMS，保留面积最大的框
        keep_bboxes = non_max_suppression_with_area(bboxes_with_areas, iou_threshold)

        # 提取保留的边界框
        filtered_bounding_boxes[cls] = [bbox for bbox, _ in keep_bboxes]

    return filtered_bounding_boxes

def add_bboxes_to_class_dict(bboxes_with_classes):
    """
    将边界框和类别信息添加到对应的类别字典中
    :param bboxes_with_classes: 边界框和类别信息的列表，格式为 [(bbox, class), ...]
    :return: 类别字典，格式为 {class: [[bbox1], [bbox2], ...]}
    """
    class_dict = {}

    for bbox1, cls1, bbox2, cls2, iou in bboxes_with_classes:
        if cls1 not in class_dict:
            class_dict[cls1] = []
        class_dict[cls1].append(bbox1)

    return class_dict

def get_labels_in_bbox(image, bbox):
    """
    获取检测框区域内包含的所有类别标签
    :param image: 图像，形状为 [224, 224]，每个像素位置存储的是类别标签
    :param bbox: 检测框，格式为 [x_min, y_min, x_max, y_max]
    :return: 检测框区域内包含的所有类别标签
    """
    # 解包检测框的坐标
    x_min, y_min, x_max, y_max = bbox

    # 提取检测框区域内的像素值
    region = image[y_min:y_max, x_min:x_max]

    # 统计检测框区域内所有唯一的类别标签
    unique_labels = np.unique(region)

    return unique_labels
    
def draw_multiple_bounding_boxes(image_path, bboxes_with_labels, colors=None):
    """
    在图像上绘制多个边界框
    :param image_path: 图像文件路径
    :param bboxes_with_labels: 边界框和类别信息的列表 [(bbox, class_label), ...]
    :param colors: 每个框的颜色列表，默认为 None（自动生成颜色）
    :return: 绘制了边界框的图像
    """
    # 读取图像
    image = cv2.imread(image_path)

    # 如果没有指定颜色，自动生成颜色
    if colors is None:
        colors = [(0, 255, 0)] * len(bboxes_with_labels.items())  # 默认绿色
    # 遍历每个边界框并绘制
    for (class_label, bbox), color in zip(bboxes_with_labels.items(), colors):
        for bboxes in bbox:
            x_min, y_min, x_max, y_max = bboxes[:]
            # 绘制矩形框
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    return image

bounding_box1 = segment_to_bouding(segmentation_label1)
filtered_bounding_boxes1 = apply_nms_to_bounding_boxes(bounding_box1, iou_threshold=0.5)
print("------------------------")
bounding_box2 = segment_to_bouding(segmentation_label2)
filtered_bounding_boxes2 = apply_nms_to_bounding_boxes(bounding_box2, iou_threshold=0.1)
print("-------------------------")
print(filtered_bounding_boxes2)
high_overlap_pairs = find_high_overlap_boxes(filtered_bounding_boxes1, filtered_bounding_boxes2, threshold=0.8)
for bbox1, cls1, bbox2, cls2, iou in high_overlap_pairs:
    print(f"bbox1: {bbox1}, cls1: {cls1}, bbox2: {bbox2}, cls2: {cls2}, iou: {iou:.4f}")
# 输出结果
overlap_dict = add_bboxes_to_class_dict(high_overlap_pairs)
image_save = draw_multiple_bounding_boxes(image_path, overlap_dict, colors=None )
cv2.imwrite("../../avsbench_data/output_image_multipleover.jpg", image_save)
unique_labels = get_labels_in_bbox(segmentation_label1, [378, 698, 546, 1279])
