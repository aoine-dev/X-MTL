import re
import json
import itertools
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import CLIPModel, AutoProcessor
from PIL import Image

from m2e2eval import f_score

multimodal_pair = Path('../data/processed_data/m2e2_annotations/article_event.json')
multimodal_gold = Path('../data/processed_data/m2e2_annotations/crossmedia_coref.txt')
m2e2_image = Path('../data/processed_data/m2e2_rawdata/image/image')
clip_path = Path('../weights/clip-vit-base-patch32')

# outputs
save_folder = Path('../outputs/m2e2-coref-CLIPScore')
result_path = save_folder / 'clip_score.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_doc_id(data_id):
    pattern = r'(.*)_(\d+)(.jpg)?'
    doc_id = re.match(pattern, data_id).group(1)
    return doc_id


class M2E2(Dataset):
    def __init__(self, path, image_dir):
        self.documents = {}
        self.documents_list = []
        self.pairs = []

        self.image_dir = image_dir
        self.id2sen = {}
        self.id2img = {}
        self.processor = AutoProcessor.from_pretrained(clip_path)

        self.load(path)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            for line in data:
                sentence_id = line['sentence_id']
                sentence = line['sentence']

                self.documents.setdefault(
                    extract_doc_id(sentence_id), [set(), set()]
                )[0].add(sentence_id)
                self.id2sen.setdefault(sentence_id, sentence)

                for image_id in line['image']:
                    image_path = self.image_dir / image_id
                    if not image_path.is_file():
                        continue

                    self.documents.setdefault(
                        extract_doc_id(image_id), [set(), set()]
                    )[1].add(image_id)
                    self.id2img.setdefault(image_id, image_path)

        for pairs in self.documents.values():
            if pairs[0] and pairs[1]:
                for sentence_id, image_id in itertools.product(*pairs):
                    self.pairs.append((sentence_id, image_id))

        self.documents_list = [[list(value[0]), list(value[1])]
                               for value in self.documents.values()]
        self.pairs = list(set(self.pairs))

    def __len__(self):
        return len(self.documents_list)

    def __getitem__(self, item):
        document = self.documents_list[item]
        texts = [self.id2sen[text] for text in document[0]]
        images = [Image.open(self.id2img[image]) for image in document[1]]
        return document, self.processor(
            texts, images, return_tensors='pt', padding=True, truncation=True, max_length=77)


class CLIPScore:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained(clip_path)

    def score(self, inputs):
        outputs = self.clip_model(**inputs)
        score = outputs.logits_per_text.tolist()
        return score


def pair2score(document, score):
    result = {}
    for i in range(len(document[0])):
        for j in range(len(document[1])):
            result[(document[0][i], document[1][j])] = score[i][j]
    return result


def save(path, score):
    with open(path, 'w', encoding='utf-8') as f:
        for (t, i), s in score.items():
            f.writelines(f'{t} {i} {s}\n')


def main():
    m2e2_document = M2E2(
        path=multimodal_pair,
        image_dir=m2e2_image
    )

    pair_scores = {}
    if result_path.is_file():
        with open(result_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                pair_scores[(line[0], line[1])] = eval(line[2])
    else:
        with tqdm(total=len(m2e2_document)) as pbar:
            score_model = CLIPScore()
            for document, pair_data in m2e2_document:
                pair_scores.update(
                    pair2score(document, score_model.score(pair_data))
                )
                pbar.update(1)
            save(result_path, pair_scores)

    # Valid
    pair_golds = []
    with open(multimodal_gold, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            pair_golds.append((line[0], line[1]))

    threshold = 0
    while threshold < 50:
        pair_predictions = [pair for pair, score in pair_scores.items() if score > threshold]
        result = f_score([pair_predictions], [pair_golds])
        print(f'threshold: {threshold}, ', end='')
        print(f'g/p: {len(pair_golds)}/{len(pair_predictions)}, ', end='')
        print('P: {0:.5f}, R: {1:.5f}, F1: {2:.5f}'.format(*result))
        threshold += 1


if __name__ == '__main__':
    main()
