# pycoco code citation: https://github.com/salaniz/pycocoevalcap/blob/master/example/coco_eval_example.py

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

class score_calculator:

    def __init__(self):

      self.annotation_file = 'captions_val2017.json'
      self.results_file = 'generated_captions.json'

    def generate_scores(self):

      # create coco object and coco_result object
      coco = COCO(self.annotation_file)
      coco_result = coco.loadRes(self.results_file)

      # create coco_eval object by taking coco and coco_result
      coco_eval = COCOEvalCap(coco, coco_result)

      # evaluate on a subset of images by setting
      # coco_eval.params['image_id'] = coco_result.getImgIds()
      # please remove this line when evaluating the full validation set
      coco_eval.params['image_id'] = coco_result.getImgIds()

      # evaluate results
      # SPICE will take a few minutes the first time, but speeds up due to caching
      coco_eval.evaluate()

      # print output evaluation scores
      for metric, score in coco_eval.eval.items():
          print(f'{metric}: {score:.3f}')

if __name__ == "__main__":
    sc = score_calculator()
    scores = sc.generate_scores()