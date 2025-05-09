import copy
import pandas as pd
from PIL import Image
import io
import os

def extract_img_from_dict(output_dir, par_df, img_hdr='image', path_key='.jpg', byte_key='bytes'):
    parquet_df = copy.deepcopy(par_df)

    for index, row in parquet_df.iterrows():
        image_info = row[img_hdr]
        if image_info and isinstance(image_info, dict):
            image_bytes = image_info.get(byte_key)
            image_path = image_info.get(path_key, f"image{index}.jpg")

            if image_bytes:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    full_path = os.path.join(output_dir, image_path)
                    image.save(full_path)
                    print(f"Saved {full_path}")
                except Exception as e:
                    print(f"Error saving image {image_path}: {e}")

def read_in_with_headers(par_fp):
    df = pd.read_parquet(par_fp)
    col_name, keys = check_bytes(df)

    return df, col_name, keys

def check_bytes(df):
    for col in df.columns:
        val = df[col].dropna().iloc[0]
        if isinstance(val, dict):
            bytes_keys = [key for key, value in val.items() if isinstance(value, (bytes, bytearray))]
            if bytes_keys:
                return col, list(val.keys())

    return None, None







