import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def make_stereo_views(img, depth, max_disp=15):
    """
    Генерирует левый и правый ракурсы (стереопару) на основе карты глубин (DIBR).
    Карта глубин нормализована от 0 (далеко) до 255 (близко).
    """
    h, w, c = img.shape
    
    # Нормализуем смещение: близкие объекты смещаются сильнее, дальние — меньше
    # Делим на 2, чтобы смещать левый ракурс влево, а правый — вправо
    disp_map = (depth.astype(np.float32) / 255.0) * (max_disp / 2.0)
    
    # Сетка координат
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Координаты со смещением
    x_left = (x - disp_map).astype(np.float32)
    x_right = (x + disp_map).astype(np.float32)
    y = y.astype(np.float32)
    
    # Переотображение пикселей с субпиксельной интерполяцией
    left_view = cv2.remap(img, x_left, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    right_view = cv2.remap(img, x_right, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return left_view, right_view


def apply_thor_olson_anaglyph(left, right):
    """
    Создает Red-Cyan анаглиф по методу Тора Олсона (Thor Olson, 2009).
    Ожидает изображения в формате BGR (стандарт OpenCV).
    """
    # Переводим в float32 и разделяем каналы (OpenCV использует BGR порядок)
    b_l, g_l, r_l = cv2.split(left.astype(np.float32))
    b_r, g_r, r_r = cv2.split(right.astype(np.float32))
    
    # Формула Тора Олсона для Red-канала (из левого глаза)
    r_out = 0.4561 * r_l + 0.5004 * g_l + 0.0435 * b_l
    
    # Формула Тора Олсона для Green-канала (из правого глаза)
    g_out = -0.0400 * r_r + 0.7380 * g_r + 0.3020 * b_r
    
    # Формула Тора Олсона для Blue-канала (из правого глаза)
    b_out = -0.0157 * r_r - 0.0157 * g_r + 1.0314 * b_r
    
    # Объединяем каналы обратно и отсекаем значения [0, 255]
    anaglyph = cv2.merge([b_out, g_out, r_out])
    anaglyph = np.clip(anaglyph, 0, 255).astype(np.uint8)
    
    return anaglyph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    # Добавляем новый ключ для анаглифа
    parser.add_argument('--anaglyph', action='store_true', help='generate Thor Olson 3D anaglyph video')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
        # Меняем логику ширины выходного видео
        if args.anaglyph or args.pred_only: 
            output_width = frame_width
        else: 
            output_width = frame_width * 2 + margin_width
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            # Получаем сырую карту глубин
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            
            # Нормализуем в диапазон [0, 255]
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_u8 = depth_normalized.astype(np.uint8)
            
            if args.anaglyph:
                # 1. Генерируем ракурсы для левого и правого глаза (max_disp можно крутить под себя)
                left_view, right_view = make_stereo_views(raw_frame, depth_u8, max_disp=16)
                # 2. Рендерим 3D по матрице Олсона
                anaglyph_frame = apply_thor_olson_anaglyph(left_view, right_view)
                out.write(anaglyph_frame)
            else:
                # Стандартное поведение исходного скрипта
                if args.grayscale:
                    depth_colored = np.repeat(depth_u8[..., np.newaxis], 3, axis=-1)
                else:
                    depth_colored = (cmap(depth_u8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                
                if args.pred_only:
                    out.write(depth_colored)
                else:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    combined_frame = cv2.hconcat([raw_frame, split_region, depth_colored])
                    out.write(combined_frame)
        
        raw_video.release()
        out.release()
