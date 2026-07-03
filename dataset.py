import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class BaseImageDataset(Dataset):
    """图像数据集基础类

    Notes:
    - second argument is `split_id` (0-based). The class will load
      `train_{split_id+1}.csv` / `val_{split_id+1}.csv` from the AGIQA-3K folder.
    - We return PIL.Image objects (resized) so the CLIPProcessor can
      perform its own normalization and tokenization. Previously the
      dataset returned a raw tensor (ToTensor) which bypassed CLIP's
      normalization and led to inconsistent preprocessing.
    """

    def __init__(self,
                 dataset_type: str,
                 split_id: int,
                 image_size: int = 224):
        super().__init__()

        self.dataset_type = dataset_type
        self.image_size = image_size
        self.is_test = dataset_type == 'test'

        # ==========================
        # 路径配置（split_id 为 0-based）
        # 使用新的数据目录：images 在 data/allimg，CSV 在 data/
        # ==========================
        base_dir = "/root/autodl-tmp/CLIP-main/data"
        self.images_dir = os.path.join(base_dir, "allimg")

        # CSV 文件可能命名为 train_{i}.csv / val_{i}.csv 或 train.csv / val.csv
        if dataset_type == 'train':
            candidate = os.path.join(base_dir, f"train_{split_id+1}.csv")
            if os.path.exists(candidate):
                self.csv_path = candidate
            else:
                fallback = os.path.join(base_dir, "train.csv")
                self.csv_path = fallback
        elif dataset_type == 'val':
            candidate = os.path.join(base_dir, f"val_{split_id+1}.csv")
            if os.path.exists(candidate):
                self.csv_path = candidate
            else:
                fallback = os.path.join(base_dir, "val.csv")
                self.csv_path = fallback
        else:
            test_path = os.path.join(base_dir, "test.csv")
            self.csv_path = test_path

        # ==========================
        # 加载数据
        # ==========================
        self._load_data()

        # ==========================
        # 图像预处理：仅做 resize，返回 PIL.Image
        # CLIPProcessor will handle normalization and conversion to tensors.
        # ==========================
        self.transform = transforms.Resize((self.image_size, self.image_size))

    def _load_data(self):
        """加载CSV数据"""
        try:
            dataInfo = pd.read_csv(self.csv_path, encoding='utf-8')

            # Ensure columns exist; accept alternative column names used in other datasets
            cols = list(dataInfo.columns)
            name_col = None
            for cand in ('name', 'image_name'):
                if cand in cols:
                    name_col = cand
                    break
            if name_col is None:
                raise KeyError(f"CSV missing required column 'name' or 'image_name' in {self.csv_path}")

            self.image_names = dataInfo[name_col].astype(str).tolist()
            # prompt may be missing or NaN in some files
            if 'prompt' in cols:
                self.prompts = dataInfo['prompt'].fillna('').astype(str).tolist()
            else:
                self.prompts = [''] * len(self.image_names)

            num_samples = len(self.image_names)
            if self.is_test:
                self.quality = [0.0] * num_samples
                self.authenticity = [0.0] * num_samples
                self.correspondence = [0.0] * num_samples
            else:
                # csv may use different column names depending on dataset source
                # Quality -> mos_quality
                quality_col = None
                for cand in ('Quality', 'mos_quality', 'quality'):
                    if cand in cols:
                        quality_col = cand
                        break
                if quality_col is not None:
                    self.quality = dataInfo[quality_col].fillna(0.0).astype(float).tolist()
                else:
                    self.quality = [0.0] * num_samples

                # Authenticity -> mos_authenticity
                auth_col = None
                for cand in ('mos_authenticity', 'Authenticity', 'authenticity'):
                    if cand in cols:
                        auth_col = cand
                        break
                if auth_col is not None:
                    self.authenticity = dataInfo[auth_col].fillna(0.0).astype(float).tolist()
                else:
                    self.authenticity = [0.0] * num_samples

                # Correspondence -> mos_align
                corr_col = None
                for cand in ('mos_align', 'Correspondence', 'correspondence'):
                    if cand in cols:
                        corr_col = cand
                        break
                if corr_col is not None:
                    self.correspondence = dataInfo[corr_col].fillna(0.0).astype(float).tolist()
                else:
                    self.correspondence = [0.0] * num_samples

        except Exception as e:
            print(f"❌ 加载数据时出错: {e}")
            raise

    def __len__(self):
        return len(self.image_names)

    def load_image(self, image_name: str) -> Image.Image:
        """加载单张图像"""
        img_path = os.path.join(self.images_dir, image_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int) -> dict:
        image_name = self.image_names[idx]
        image = self.load_image(image_name)
        # 保持为 PIL.Image（或已 resize 的 PIL），CLIPProcessor 会处理为 tensor
        image_resized = self.transform(image)

        data = {
            "prompt": self.prompts[idx],
            "name": image_name,
            # return PIL.Image so downstream CLIPProcessor can handle normalization
            "image": image_resized
        }

        if not self.is_test:
            data.update({
                "Quality": self.quality[idx],
                "Authenticity": self.authenticity[idx],
                "Correspondence": self.correspondence[idx]
            })

        return data


# ==========================
# 封装不同模式（统一接口）
# ==========================
class DatasetImage(BaseImageDataset):
    """图像数据集（统一接口，兼容原命名）

    Note: keep the same calling convention used by the training script:
    DatasetImage('train', split_id)
    where split_id is 0-based.
    """
    def __init__(self, dataset_type: str, split_id: int, image_size: int = 224):
        super().__init__(dataset_type, split_id, image_size=image_size)
