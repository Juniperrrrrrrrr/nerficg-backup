import cv2
import os
import glob

# 路径设置
input_dir = '/home/bupt803/桌面/home/nerficg/dataset/lab/images'
output_dir = '/home/bupt803/桌面/home/nerficg/dataset/lab_half/images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 使用你改进的匹配逻辑
image_files = (
    glob.glob(os.path.join(input_dir, '*.jpg')) +
    glob.glob(os.path.join(input_dir, '*.JPG')) +
    glob.glob(os.path.join(input_dir, '*.png')) +
    glob.glob(os.path.join(input_dir, '*.PNG')) +
    glob.glob(os.path.join(input_dir, '*.jpeg'))
)

# 去重（防止某些系统下重复匹配）
image_files = list(set(image_files))

print(f"找到 {len(image_files)} 张图片，准备进行二倍下采样...")

for f in image_files:
    img = cv2.imread(f)
    if img is None:
        print(f"警告: 无法读取文件 {f}")
        continue
    
    # 获取文件名和后缀
    basename = os.path.basename(f)
    ext = os.path.splitext(basename)[1].lower() # 获取小写后缀
    
    # 计算下采样后的尺寸
    new_size = (img.shape[1] // 2, img.shape[0] // 2)
    
    # 缩放 (INTER_AREA 适合缩小图片，画质最稳)
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # 2. 根据原图格式选择保存参数
    save_path = os.path.join(output_dir, basename)
    
    if ext in ['.jpg', '.jpeg']:
        # JPG 保持高画质 95
        cv2.imwrite(save_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == '.png':
        # PNG 使用压缩级别 3 (较快且无损)
        cv2.imwrite(save_path, resized_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        # 其他格式直接保存
        cv2.imwrite(save_path, resized_img)
        
    print(f"成功处理: {basename} -> {new_size}")

print(f"\n全部处理完成！图片保存在: {output_dir}")