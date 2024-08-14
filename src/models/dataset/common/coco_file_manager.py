from typing import List, Dict, Optional
from collections import defaultdict

from picsellia_annotations.coco import COCOFile, Image, Annotation


class COCOFileManager:
    def __init__(self, coco_file: COCOFile):
        self.coco_file = coco_file
        self._build_indices()

    def _build_indices(self):
        """Construit des index pour accélérer les recherches."""
        self.category_id_to_name = {
            cat.id: cat.name for cat in self.coco_file.categories
        }
        self.category_name_to_id = {
            cat.name: cat.id for cat in self.coco_file.categories
        }
        self.image_id_to_filename = {
            img.id: img.file_name for img in self.coco_file.images
        }
        self.filename_to_image_id = {
            img.file_name: img.id for img in self.coco_file.images
        }

        self.image_id_to_annotations = defaultdict(list)
        for ann in self.coco_file.annotations:
            self.image_id_to_annotations[ann.image_id].append(ann)

    def get_category_name(self, category_id: int) -> Optional[str]:
        """Retourne le nom de la catégorie pour un ID donné."""
        return self.category_id_to_name.get(category_id)

    def get_category_id(self, category_name: str) -> Optional[int]:
        """Retourne l'ID de la catégorie pour un nom donné."""
        return self.category_name_to_id.get(category_name)

    def get_image_filename(self, image_id: int) -> Optional[str]:
        """Retourne le nom de fichier de l'image pour un ID donné."""
        return self.image_id_to_filename.get(image_id)

    def get_image_id(self, filename: str) -> Optional[int]:
        """Retourne l'ID de l'image pour un nom de fichier donné."""
        return self.filename_to_image_id.get(filename)

    def get_annotations_for_image(self, image_id: int) -> List[Annotation]:
        """Retourne toutes les annotations pour une image donnée."""
        return self.image_id_to_annotations.get(image_id, [])

    def get_images_for_category(self, category_id: int) -> List[Image]:
        """Retourne toutes les images contenant une catégorie donnée."""
        image_ids = set(
            ann.image_id
            for ann in self.coco_file.annotations
            if ann.category_id == category_id
        )
        return [img for img in self.coco_file.images if img.id in image_ids]

    def get_annotation_count_per_category(self) -> Dict[str, int]:
        """Retourne le nombre d'annotations par catégorie."""
        count: defaultdict[str, int] = defaultdict(int)
        for ann in self.coco_file.annotations:
            category_name = self.get_category_name(ann.category_id)
            if category_name is not None:
                count[category_name] += 1
        return dict(count)

    def get_image_dimensions(self, image_id: int) -> Optional[Dict[str, int]]:
        """Retourne les dimensions d'une image donnée."""
        for img in self.coco_file.images:
            if img.id == image_id:
                return {"width": img.width, "height": img.height}
        return None

    def get_annotations_by_category(self, category_id: int) -> List[Annotation]:
        """Retourne toutes les annotations pour une catégorie donnée."""
        return [
            ann for ann in self.coco_file.annotations if ann.category_id == category_id
        ]
