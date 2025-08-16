import json
from pathlib import Path
from tqdm import tqdm

from stanfordcorenlp import StanfordCoreNLP
from ultralytics import YOLO

corenlp_path = '../../../weights/stanford-corenlp-full-2018-10-05'
yolo_path = '../../../weights/yolov8x.pt'

# Input
voa_path = Path('../../processed_data/voa')
voa_text_path = voa_path / 'voa_img_dataset.json'
voa_image_path = voa_path / 'image'

# Output
voa_text_entity_path = voa_path / 'voa_text_entity_StanfordNER.json'
voa_image_entity_path = voa_path / 'voa_image_entity_YOLOv8.json'


class VOAPair:
    def __init__(self, path, image_dir):
        with open(path, 'r') as f:
            self.raw_data = json.load(f)

        self.sentence_ids = []
        self.sentences = []
        self.image_ids = []
        self.image_paths = []

        for doc_id, line in tqdm(self.raw_data.items(), desc='Load VOA'):
            for frag_id, pair in line.items():
                image_path = image_dir / f'{doc_id}_{frag_id}.jpg'
                if not image_path.exists():
                    continue

                caption = pair['cap'].strip().replace('\u200b', '').replace('\ufeff', '')
                if not caption:
                    continue

                self.sentence_ids.append(f'{doc_id}_{frag_id}')
                self.sentences.append(caption)
                self.image_ids.append(f'{doc_id}_{frag_id}.jpg')
                self.image_paths.append(image_path)


class StanfordNER:
    def __init__(self, path):
        self.nlp = StanfordCoreNLP(path, lang='en')

    def extract(self, sentence):
        res = self.nlp.ner(sentence)

        words, tags = [], []
        for i in range(len(res)):
            word, tag = res[i]
            words.append(word)
            if tag != 'O':
                tags.append((word, tag, i))

        entities = []
        stack = []
        for tag in tags:
            if not stack:
                stack.append(tag)
            else:
                if stack[-1][1] == tag[1] and stack[-1][2] + 1 == tag[2]:
                    stack.append(tag)
                else:
                    entities.append((stack[0][1], stack[0][2], stack[0][2] + len(stack)))
                    stack = [tag]
        if stack:
            entities.append((stack[0][1], stack[0][2], stack[0][2] + len(stack)))

        return words, entities


class YOLODetection:
    def __init__(self, weights):
        self.model = YOLO(weights)
        self.names = self.model.names

    def run(self, path):
        results = []
        for result in self.model(path, verbose=False):
            boxes = result.boxes.cpu()
            for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                xyxy = tuple(xyxy.tolist())
                conf = float(conf)
                cls = self.names[int(cls)]
                results.append([xyxy, conf, cls])
        return results


if __name__ == '__main__':
    voa_data = VOAPair(voa_text_path, voa_image_path)

    # Named Entity Recognition
    ner = StanfordNER(corenlp_path)
    voa_text_entity = {}
    for sentence_id, sentence in tqdm(
            zip(voa_data.sentence_ids, voa_data.sentences),
            total=len(voa_data.sentence_ids), desc='StanfordNER'):
        word, entity = ner.extract(sentence)
        voa_text_entity[sentence_id] = {
            'words': word,
            'golden-entity-mentions': [{
                'text': ' '.join(word[e[1]:e[2]]),
                'entity-type': e[0],
                'start': e[1],
                'end': e[2]
            } for e in entity]
        }
    with open(voa_text_entity_path, 'w', encoding='utf-8') as f:
        json.dump(voa_text_entity, f, indent=4)

    # Object Detection
    od = YOLODetection(yolo_path)
    voa_image_entity = {}
    for image_id, image_path in tqdm(
            zip(voa_data.image_ids, voa_data.image_paths),
            total=len(voa_data.image_ids), desc='YOLOv8'):
        voa_image_entity[image_id] = od.run(image_path)
    with open(voa_image_entity_path, 'w', encoding='utf-8') as f:
        json.dump(voa_image_entity, f, indent=4)
