import os
import io
import argparse
from PIL import Image
from PIL.ExifTags import TAGS

def extract_focal_length_from_exif(image_path):
    """
    Пытается извлечь реальное и эквивалентное фокусное расстояние из EXIF
    и возвращает нормализованное фокусное расстояние согласно спецификации Google.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()
            if not exif_data:
                return None

            # Подтягиваем суб-директорию EXIF (там лежат теги фокуса)
            # 0x8769 - это стандартный указатель на Exif IFD
            exif_ifd = exif_data.get_ifd(0x8769)

            focal_mm = None
            focal_35mm = None

            # Ищем нужные теги
            for tag_id, value in exif_ifd.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "FocalLength":
                    if isinstance(value, tuple) and value != 0:
                        focal_mm = value[0] / value[1]
                    else:
                        focal_mm = float(value)
                elif tag_name == "FocalLengthIn35mmFilm":
                    focal_35mm = float(value)

            if focal_mm and focal_35mm:
                normalized_focal = focal_mm / focal_35mm
                print(f"[EXIF] Найдено FocalLength: {focal_mm}mm, 35mm-Эквивалент: {focal_35mm}mm.")
                print(f"[EXIF] Рассчитано нормализованное фокусное расстояние: {normalized_focal:.6f}")
                return normalized_focal
    except Exception as e:
        print(f"[EXIF] Предупреждение: Не удалось прочесть данные фокуса из EXIF ({e})")

    return None

def process_and_crop_depth_map(depth_path, target_aspect):
    """
    Открывает карту глубин, кадрирует её по центру под целевой aspect ratio 
    и возвращает бинарные данные PNG из буфера памяти.
    """
    with Image.open(depth_path) as depth_img:
        # Принудительно конвертируем в Grayscale (L), если карта глубин цветная
        if depth_img.mode not in ("L", "I;16"):
            depth_img = depth_img.convert("L")

        dw, dh = depth_img.size
        current_aspect = dw / dh

        # Проверяем несовпадение пропорций с допуском на округление
        if abs(current_aspect - target_aspect) > 0.001:
            print(f"[Обработка] Пропорции карты глубин ({current_aspect:.4f}) не совпадают с фото ({target_aspect:.4f}).")
            if current_aspect > target_aspect:
                # Карта глубин слишком широкая — обрезаем бока
                new_dw = int(dh * target_aspect)
                left = (dw - new_dw) // 2
                right = left + new_dw
                top = 0
                bottom = dh
            else:
                # Карта глубин слишком высокая — обрезаем верх и низ
                new_dh = int(dw / target_aspect)
                left = 0
                right = dw
                top = (dh - new_dh) // 2
                bottom = top + new_dh

            print(f"[Обработка] Обрезка карты глубин по центру: с {dw}x{dh} до {right-left}x{bottom-top}")
            depth_img = depth_img.crop((left, top, right, bottom))
        else:
            print("[Обработка] Пропорции карты глубин уже идеально соответствуют фотографии.")

        # Сохраняем результат в байтовый буфер (в памяти)
        img_byte_arr = io.BytesIO()
        depth_img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

def create_dynamic_depth_jpeg(
    primary_image_path, 
    depth_map_path, 
    output_path, 
    depth_format, 
    near, 
    far,
    focal_length_px=None,
    principal_point_x=0.5,
    principal_point_y=0.5
):
    """
    Собирает валидный Google Dynamic Depth 1.0 файл с автоматической подготовкой карты глубин.
    """
    # 1. Анализируем размеры оригинальной фотографии
    try:
        with Image.open(primary_image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"Ошибка при открытии основного изображения: {e}")
        return

    img_aspect = img_w / img_h
    print(f"[Геометрия] Фотография: {img_w}x{img_h} (Aspect Ratio: {img_aspect:.4f})")

    # 2. Обрабатываем и кадрируем карту глубин в памяти
    if not os.path.exists(depth_map_path):
        print(f"Ошибка: Файл карты глубин не найден: {depth_map_path}")
        return

    depth_data = process_and_crop_depth_map(depth_map_path, img_aspect)
    depth_length = len(depth_data)

    # 3. Расчет калибровки камеры (ImagingModel)
    exif_focal = extract_focal_length_from_exif(primary_image_path)

    if exif_focal is not None:
        focal_x = exif_focal
        focal_y = exif_focal
    elif focal_length_px is not None:
        max_dim = max(img_w, img_h)
        focal_x = focal_length_px / max_dim
        focal_y = focal_length_px / max_dim
    else:
        print("[Калибровка] Данные фокуса не найдены ни в EXIF, ни в аргументах. Используется 1.0.")
        focal_x = 1.0
        focal_y = 1.0

    # 4. Формируем XML-метаданные XMP
    xmp_meta = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, 2017/05/06-01:02:15        ">
 <rdf:RDF xmlns:rdf="http://w3.org">
  
  <rdf:Description rdf:about=""
    xmlns:Device="http://google.com"
    xmlns:Profile="http://google.com">
   <Device:Profiles>
    <rdf:Seq>
     <rdf:li>
      <rdf:Description Profile:Type="DepthPhoto">
       <Profile:CameraIndices>
        <rdf:Seq>
         <rdf:li>0</rdf:li>
        </rdf:Seq>
       </Profile:CameraIndices>
      </rdf:Description>
     </rdf:li>
    </rdf:Seq>
   </Device:Profiles>
  </rdf:Description>

  <rdf:Description rdf:about=""
    xmlns:Container="http://google.com">
   <Container:Directory>
    <rdf:Seq>
     <rdf:li>
      <rdf:Description Container:Mime="image/jpeg">
       <Container:Length>0</Container:Length>
      </rdf:Description>
     </rdf:li>
     <rdf:li>
      <rdf:Description Container:Mime="image/png">
       <Container:Length>{depth_length}</Container:Length>
       <Container:DataURI>android/depthmap</Container:DataURI>
      </rdf:Description>
     </rdf:li>
    </rdf:Seq>
   </Container:Directory>
  </rdf:Description>

  <rdf:Description rdf:about=""
    xmlns:Camera="http://google.com"
    xmlns:DepthMap="http://google.com"
    xmlns:ImagingModel="http://google.com">
   
   <Camera:Image>
    <rdf:Description Camera:ItemSemantic="Primary" Camera:ItemURI="android/primary" />
   </Camera:Image>
   
   <Camera:DepthMap>
    <rdf:Description 
      DepthMap:Format="{depth_format}"
      DepthMap:Near="{near}"
      DepthMap:Far="{far}"
      DepthMap:Units="Meters"
      DepthMap:DepthURI="android/depthmap"/>
   </Camera:DepthMap>

   <Camera:ImagingModel>
    <rdf:Description 
      ImagingModel:ImageWidth="{img_w}"
      ImagingModel:ImageHeight="{img_h}"
      ImagingModel:FocalLengthX="{focal_x:.6f}"
      ImagingModel:FocalLengthY="{focal_y:.6f}"
      ImagingModel:PrincipalPointX="{principal_point_x:.6f}"
      ImagingModel:PrincipalPointY="{principal_point_y:.6f}"
      ImagingModel:PixelAspectRatio="1.0"
      ImagingModel:Skew="0.0"/>
   </Camera:ImagingModel>
  </rdf:Description>

 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''.encode('utf-8')

    # 5. Читаем исходный JPEG и проверяем маркер SOI
    with open(primary_image_path, 'rb') as f:
        jpeg_data = f.read()

    if not jpeg_data.startswith(b'\xFF\xD8'):
        print("Ошибка: Исходный файл фотографии не является валидным JPEG.")
        return

    # Заворачиваем XMP в маркер APP1 (0xFFE1)
    xmp_header = b'http://adobe.com\x00'
    payload = xmp_header + xmp_meta
    length_bytes = (len(payload) + 2).to_bytes(2, byteorder='big')
    app1_marker = b'\xFF\xE1' + length_bytes + payload

    # 6. Запись результирующего файла
    with open(output_path, 'wb') as f:
        f.write(jpeg_data[:2])          
        f.write(app1_marker)            
        f.write(jpeg_data[2:])          
        f.write(depth_data)             
    
    print(f"Сборка завершена успешно! Результат сохранен в: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Сборщик файлов формата Google Dynamic Depth 1.0 с авто-кропом карты глубин и EXIF."
    )
    
    parser.add_argument("photo", help="Путь к исходной фотографии (JPEG)")
    parser.add_argument("depth", help="Путь к карте глубин (PNG)")
    parser.add_argument("output", help="Путь для сохранения итогового файла (JPEG)")
    
    parser.add_argument(
        "--format", 
        choices=["RangeLinear", "RangeInverse"], 
        default="RangeLinear",
        help="Формат кодирования карты глубин (по умолчанию: RangeLinear)"
    )
    parser.add_argument(
        "--near", 
        type=float, 
        default=0.5,
        help="Минимальная дистанция съемки в метрах (по умолчанию: 0.5)"
    )
    parser.add_argument(
        "--far", 
        type=float, 
        default=10.0,
        help="Максимальная дистанция съемки в метрах (по умолчанию: 10.0)"
    )
    
    parser.add_argument(
        "--focal", 
        type=float, 
        metavar="PX",
        help="Резервное фокусное расстояние в пикселях (используется, только если в EXIF нет данных)."
    )
    parser.add_argument(
        "--ppx", 
        type=float, 
        default=0.5,
        help="Координата X оптического центра (нормализованная от 0.0 до 1.0, по умолчанию: 0.5)"
    )
    parser.add_argument(
        "--ppy", 
        type=float, 
        default=0.5,
        help="Координата Y оптического центра (нормализованная от 0.0 до 1.0, по умолчанию: 0.5)"
    )

    args = parser.parse_args()

    create_dynamic_depth_jpeg(
	primary_image_path=args.photo,
	depth_map_path=args.depth,
	output_path=args.output,
	depth_format=args.format,
	near=args.near,
	far=args.far,
	focal_length_px=args.focal,
	principal_point_x=args.ppx,
	principal_point_y=args.ppy
    )
