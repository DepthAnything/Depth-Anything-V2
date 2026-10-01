import io
import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Optional

class GoogleDynamicDepthGenerator:
    """
    Класс для генерации изображений в формате Google Dynamic Depth 1.0.
    Объединяет основное изображение и карту глубин (OpenCV) в один JPEG-файл с XMP-метаданными.
    """
    def __init__(self, primary_image_path: str, depth_map: np.ndarray):
        """
        Инициализация генератора.
        
        :param primary_image_path: Путь к исходному изображению (.jpeg, .jpg, .png)
        :param depth_map: Карта глубин в формате cv2 (numpy array)
        """
        if not os.path.exists(primary_image_path):
            raise FileNotFoundError(f"Основное изображение не найдено: {primary_image_path}")
        
        if not isinstance(depth_map, np.ndarray):
            raise TypeError("Карта глубин должна быть массивом numpy (cv2 image).")

        self.primary_image_path = primary_image_path
        self._depth_map = depth_map

        # Настройки по умолчанию (свойства)
        self.depth_format = "RangeLinear"  # Опции: RangeLinear, RangeInverse
        self.near = 0.5                     # Ближняя граница в метрах
        self.far = 10.0                    # Дальняя граница в метрах
        self.focal_length_px = None        # Фокусное расстояние в пикселях (резервное)
        self.principal_point_x = 0.5       # Оптический центр X (нормализованный)
        self.principal_point_y = 0.5       # Оптический центр Y (нормализованный)

    def _extract_focal_length_from_exif(self) -> Optional[float]:
        """Извлекает нормализованное фокусное расстояние из EXIF-данных."""
        try:
            with Image.open(self.primary_image_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return None

                # Стандартный указатель на Exif IFD (0x8769)
                exif_ifd = exif_data.get_ifd(0x8769)
                
                focal_mm = None
                focal_35mm = None

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
                    return focal_mm / focal_35mm
        except Exception as e:
            print(f"[EXIF] Предупреждение: Не удалось прочесть EXIF данные фокуса ({e})")
        return None

    def _process_depth_map(self, target_aspect: float) -> bytes:
        """
        Приводит карту глубин к Grayscale, кадрирует по центру 
        под целевой aspect ratio и возвращает байты PNG.
        """
        # Превращаем в оттенки серого, если карта трехканальная (BGR)
        if len(self._depth_map.shape) == 3:
            depth_gray = cv2.cvtColor(self._depth_map, cv2.COLOR_BGR2GRAY)
        else:
            depth_gray = self._depth_map.copy()

        dh, dw = depth_gray.shape[:2]
        current_aspect = dw / dh

        # Центрированное кадрирование при несовпадении пропорций
        if abs(current_aspect - target_aspect) > 0.001:
            if current_aspect > target_aspect:
                new_dw = int(dh * target_aspect)
                left = (dw - new_dw) // 2
                depth_gray = depth_gray[:, left:left + new_dw]
            else:
                new_dh = int(dw / target_aspect)
                top = (dh - new_dh) // 2
                depth_gray = depth_gray[top:top + new_dh, :]

        # Кодируем изображение cv2 в PNG в буфер памяти
        success, encoded_img = cv2.imencode('.png', depth_gray)
        if not success:
            raise RuntimeError("Не удалось закодировать карту глубин в PNG")
            
        return encoded_img.tobytes()

    def _get_primary_jpeg_bytes(self) -> bytes:
        """Конвертирует исходный файл в JPEG байты, если это PNG, или читает как есть."""
        if self.primary_image_path.lower().endswith(('.jpg', '.jpeg')):
            with open(self.primary_image_path, 'rb') as f:
                return f.read()
        else:
            # Если исходный файл PNG/BMP и т.д., конвертируем его в JPEG в памяти
            with Image.open(self.primary_image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=95)
                return buffer.getvalue()

    def generate(self, output_path: str):
        """
        Собирает финальный файл Google Dynamic Depth JPEG и записывает на диск.
        
        :param output_path: Путь для сохранения результирующего .jpg/.jpeg файла
        """
        # 1. Получаем размеры исходного фото
        with Image.open(self.primary_image_path) as img:
            img_w, img_h = img.size
        img_aspect = img_w / img_h

        # 2. Обрабатываем массив карты глубин
        depth_data = self._process_depth_map(img_aspect)
        depth_length = len(depth_data)

        # 3. Рассчитываем параметры фокуса (ImagingModel)
        exif_focal = self._extract_focal_length_from_exif()
        if exif_focal is not None:
            focal_x = focal_y = exif_focal
        elif self.focal_length_px is not None:
            max_dim = max(img_w, img_h)
            focal_x = focal_y = self.focal_length_px / max_dim
        else:
            focal_x = focal_y = 1.0

        # 4. Формируем XML-метаданные XMP
        xmp_meta = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, 2017/05/06-01:02:15        ">
 <rdf:RDF xmlns:rdf="http://w3.org">
  <rdf:Description rdf:about="" xmlns:Device="http://google.com" xmlns:Profile="http://google.com">
   <Device:Profiles>
    <rdf:Seq>
     <rdf:li>
      <rdf:Description Profile:Type="DepthPhoto">
       <Profile:CameraIndices><rdf:Seq><rdf:li>0</rdf:li></rdf:Seq></Profile:CameraIndices>
      </rdf:Description>
     </rdf:li>
    </rdf:Seq>
   </Device:Profiles>
  </rdf:Description>
  <rdf:Description rdf:about="" xmlns:Container="http://google.com">
   <Container:Directory>
    <rdf:Seq>
     <rdf:li><rdf:Description Container:Mime="image/jpeg"><Container:Length>0</Container:Length></rdf:Description></rdf:li>
     <rdf:li>
      <rdf:Description Container:Mime="image/png">
       <Container:Length>{depth_length}</Container:Length>
       <Container:DataURI>android/depthmap</Container:DataURI>
      </rdf:Description>
     </rdf:li>
    </rdf:Seq>
   </Container:Directory>
  </rdf:Description>
  <rdf:Description rdf:about="" xmlns:Camera="http://google.com" xmlns:DepthMap="http://google.com" xmlns:ImagingModel="http://google.com">
   <Camera:Image><rdf:Description Camera:ItemSemantic="Primary" Camera:ItemURI="android/primary" /></Camera:Image>
   <Camera:DepthMap>
    <rdf:Description DepthMap:Format="{self.depth_format}" DepthMap:Near="{self.near}" DepthMap:Far="{self.far}" DepthMap:Units="Meters" DepthMap:DepthURI="android/depthmap"/>
   </Camera:DepthMap>
   <Camera:ImagingModel>
    <rdf:Description ImagingModel:ImageWidth="{img_w}" ImagingModel:ImageHeight="{img_h}" ImagingModel:FocalLengthX="{focal_x:.6f}" ImagingModel:FocalLengthY="{focal_y:.6f}" ImagingModel:PrincipalPointX="{self.principal_point_x:.6f}" ImagingModel:PrincipalPointY="{self.principal_point_y:.6f}" ImagingModel:PixelAspectRatio="1.0" ImagingModel:Skew="0.0"/>
   </Camera:ImagingModel>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''.encode('utf-8')

        # 5. Подготавливаем базовые JPEG байты
        jpeg_data = self._get_primary_jpeg_bytes()

        # Заворачиваем XMP в маркер APP1 (0xFFE1)
        xmp_header = b'http://adobe.com\x00'
        payload = xmp_header + xmp_meta
        length_bytes = (len(payload) + 2).to_bytes(2, byteorder='big')
        app1_marker = b'\xFF\xE1' + length_bytes + payload

        # 6. Склеиваем файл: SOI + APP1 маркер + Оставшаяся часть JPEG + Бинарник карты глубин
        with open(output_path, 'wb') as f:
            f.write(jpeg_data[:2])          # Записываем \xFF\xD8 (SOI)
            f.write(app1_marker)            # Внедряем метаданные
            f.write(jpeg_data[2:])          # Дописываем оригинальный JPEG контент
            f.write(depth_data)             # В самый конец прикрепляем PNG карты глубин
        
