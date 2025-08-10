import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


class CocoDataset:

    def __init__(self, data_dir, data_type, seed=32, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        ann_fp = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        self.coco = COCO(annotation_file=ann_fp)
        self.random = np.random.RandomState(seed=seed)
        # COCO is a standard dataset for instance segmentation. Following AlignSAM, target categories
        # are sampled and the largest instance annotation is used to define the target mask for the
        # interactive episode. This encourages meaningful, non-trivial targets.

    def configure_targets(self, target_categories):
        try:
            target_cat_ids = self.coco.getCatIds(catNms=target_categories)
        except:
            cats = self.coco.loadCats(self.coco.getCatIds())
            cat_names =[cat['name'] for cat in cats]
            raise Exception("Incorrect categories passed"
                                ". Choose from the following: {}".format(cat_names))
        
        return target_cat_ids, self.coco.getImgIds(catIds=target_cat_ids)

    def load_image(self, img_id):
        img = self.coco.loadImgs([img_id])[0]

        img_filename = img['file_name']
        img_fp = '{}/images/{}/{}'.format(self.data_dir, self.data_type, img_filename)

        if not os.path.exists(img_fp):
            print(f"Image file not found: {img_fp}, data_dir: {self.data_dir}, data_type: {self.data_type}, img_filename: {img_filename}, img_id: {img_id}")
            raise FileNotFoundError()
        
        return cv2.imread(img_fp, -1)

    def get_sample(self, target_categories):
        target_cat_ids, img_ids = self.configure_targets(target_categories)

        while True:
            sample_img_id = self.random.choice(img_ids)
            sample_img = self.load_image(sample_img_id)
            if len(sample_img.shape) == 2:
                print(f"Wrong shape: {sample_img.shape}, sample_img_id: {sample_img_id}, img_ids: {img_ids}")
                sample_img = sample_img.reshape(sample_img.shape[0], sample_img.shape[1], 1)
            h,w,_ = sample_img.shape

            ann_ids = self.coco.getAnnIds(imgIds=sample_img_id, 
                                        catIds=target_cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            anns = sorted(anns, key=lambda x: x['area'], reverse=True)
            sample_mask = self.coco.annToMask(anns[0])

            # Ensure the largest annotation is significant enough to be considered a valid sample.
            # Heuristic mirrors AlignSAM-style curation: avoid tiny instances that provide little
            # informative signal for interactive segmentation.
            if anns[0]['area'] > 0.01 * h * w:
                break

            # sample_mask = np.zeros((h,w), dtype=np.uint8)
            # for ann in anns:
            #     mask = self.coco.annToMask(ann)
            #     sample_mask[mask == 1] = target_cat_ids.index(ann['category_id']) + 1

        return sample_img, sample_mask

    
if __name__ == "__main__":
    data_dir = '/Users/shantanu/Datasets/coco-dataset'
    data_type = 'val2017'
    target_categories = ['person', 'cat']

    coco_dataset = CocoDataset(data_dir, data_type, seed=5122)

    fig = plt.figure()
    ax = plt.gca()

    while True:
        img, mask = coco_dataset.get_sample(target_categories)
        mask_scale = int(255 / len(target_categories))

        mask_color = cv2.applyColorMap(mask * mask_scale, colormap=cv2.COLORMAP_MAGMA)
        output_img = np.hstack([img, mask_color])
        
        ax.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        fig.canvas.draw()
        
        plt.waitforbuttonpress()
