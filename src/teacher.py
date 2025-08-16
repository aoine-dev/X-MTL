import os
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from loguru import logger

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from transformers import BertModel, CLIPVisionModel, AutoTokenizer, AutoProcessor
from PIL import Image

from m2e2eval import load_json, iou, f_score, acc_score, f_score_iou, refine_result, eval_all

# Dataset Path
ace_path = Path('../data/processed_data/ace05-en')
ace_train = ace_path / 'JMEE_train_filter_no_timevalue.json'
ace_dev = ace_path / 'JMEE_dev_filter_no_timevalue.json'
ace_test = ace_path / 'JMEE_test_filter_no_timevalue.json'

imSitu_path = Path('../data/processed_data/imSitu')
swig_path = Path('../data/processed_data/SWiG')
imSitu_train = imSitu_path / 'train.json'
imSitu_dev = imSitu_path / 'dev.json'
imSitu_test = imSitu_path / 'test.json'
swig_train = swig_path / 'train.json'
swig_dev = swig_path / 'dev.json'
swig_test = swig_path / 'test.json'
imSitu_image = imSitu_path / 'of500_images_resized'

m2e2_path = Path('../data/processed_data/m2e2_annotations')
m2e2_text_test = m2e2_path / 'article_event.json'
m2e2_image_test = m2e2_path / 'image_event.json'
m2e2_object = m2e2_path / 'm2e2_YOLOv8.json'
m2e2_schema = m2e2_path / 'ace_sr_mapping.txt'
m2e2_image_path = Path('../data/processed_data/m2e2_rawdata/image/image')

# Outputs
save_folder = Path('../outputs/ace05-imsitu_event_XMTL_sep')
save_folder.mkdir(parents=True, exist_ok=True)
# log_path = save_folder / f'log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt'
log_path = save_folder / 'log_step_train_sep.txt'
checkpoint_path = save_folder / 'checkpoint_XMTL.pkl'
text_event_pred_path = save_folder / 'predictions_m2e2_text_event.json'
text_arg_pred_path = save_folder / 'predictions_m2e2_text_arg.json'
image_event_pred_path = save_folder / 'predictions_m2e2_image_event.json'
image_arg_pred_path = save_folder / 'predictions_m2e2_image_arg.json'

# Pretrained Model Path
bert_path = Path('../weights/bert-base-uncased')
clip_path = Path('../weights/clip-vit-base-patch32')

# max_length = 200
batch_size = 64
lr_list = [(1e-5, 1e-6, 1e-4, False, 'tee'), (1e-5, 1e-6, 1e-4, False, 'vee'),
           (1e-5, 1e-6, 1e-4, False, 'tae'), (1e-5, 1e-6, 1e-4, False, 'vae')]
epoch = 10
seed = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class M2E2Schema:
    def __init__(self, path, add_none=True):
        self.mapping = {}
        with open(path) as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                verb = line[0]
                v_role = line[1]
                event = line[2].replace('||', ':').replace('|', '-')
                if event == 'Transaction:Transfer-MONEY':
                    event = 'Transaction:Transfer-Money'
                e_role = line[3]
                self.mapping[(verb, v_role)] = (event, e_role)

        self.verbs = sorted(list(set(key[0] for key in self.mapping.keys())))
        self.v_roles = sorted(list(set(key[1] for key in self.mapping.keys())))
        self.events = sorted(list(set(value[0] for value in self.mapping.values())))
        self.e_roles = sorted(list(set(value[1] for value in self.mapping.values())))

        self.verb2event = {key[0]: value[0] for key, value in self.mapping.items()}
        self.event2role = {}
        for event, role in self.mapping.values():
            self.event2role.setdefault(event, set()).add(role)

        if add_none:
            self.verbs = ['None'] + self.verbs
            self.v_roles = ['None'] + self.v_roles
            self.events = ['None'] + self.events
            self.e_roles = ['None'] + self.e_roles


class BIO:
    def __init__(self, tags):
        self.tags = tags
        self.empty_tag = 'O'

        self.index2tag = {0: self.empty_tag}
        self.tag2index = {self.empty_tag: 0}
        for i, item in enumerate(tags):
            self.index2tag[2 * i + 1] = f'B-{item}'
            self.index2tag[2 * i + 2] = f'I-{item}'

            self.tag2index[f'B-{item}'] = 2 * i + 1
            self.tag2index[f'I-{item}'] = 2 * i + 2

    def encode(self, seq, tag_offset):
        """
        Encode tags and offsets into seqs.
        """
        for tag_type, start, end in tag_offset:
            tag = [f'B-{tag_type}'] + [f'I-{tag_type}'] * (end - start - 1)
            seq[start:end] = tag
        return seq

    def decode(self, seq):
        """
        Extract tags and offsets from seqs.
        """
        if isinstance(seq[0], str):
            seq = self.t2i(seq)

        tag_offsets = []
        b_indexes = [index for index, item in enumerate(seq) if item % 2]
        for i in b_indexes:
            for j in range(i + 1, len(seq)):
                if seq[j] != (seq[i] + 1):
                    tag_offsets.append((self.tags[(seq[i] - 1) // 2], i, j))
                    break
        return tag_offsets

    def i2t(self, seq):
        return [self.index2tag[item] for item in seq]

    def t2i(self, seq):
        return [self.tag2index[item] for item in seq]

    def encode_batch(self, seqs, tag_offsets):
        return [self.encode(seq, tag_offset) for seq, tag_offset in zip(seqs, tag_offsets)]

    def decode_batch(self, seqs):
        return [self.decode(seq) for seq in seqs]

    def i2t_batch(self, seqs):
        return [self.i2t(seq) for seq in seqs]

    def t2i_batch(self, seqs):
        return [self.t2i(seq) for seq in seqs]


def image_prompt(image_path, bbox, origin_size=None, target_size=(512, 512),
                 save_path=None, is_show=False):
    image = cv2.imread(str(image_path))
    if not origin_size:
        origin_size = (image.shape[0], image.shape[1])
    if (image.shape[0], image.shape[1]) != target_size:
        image = cv2.resize(image, target_size)

    x0 = int(bbox[0] * image.shape[1] / origin_size[1])
    y0 = int(bbox[1] * image.shape[0] / origin_size[0])
    x1 = int(bbox[2] * image.shape[1] / origin_size[1])
    y1 = int(bbox[3] * image.shape[0] / origin_size[0])
    x_center = int((x0 + x1) / 2)
    y_center = int((y0 + y1) / 2)
    radius = int(pow(pow(x1 - x0, 2) + pow(y1 - y0, 2), 0.5) / 2)

    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    # cv2.circle(image, (x_center, y_center), radius, (0, 0, 255), 2)

    if save_path:
        cv2.imwrite(str(save_path), image)

    if is_show:
        cv2.imshow('Image Prompt', image)
        cv2.waitKey(0)


def build_prompt(image_paths, bboxes_with_size, target_dir):
    image_prompt_paths = []
    for i in tqdm(range(len(image_paths)), desc='Build Image Prompt'):
        image_path = image_paths[i]
        bbox, image_size = bboxes_with_size[i]

        image_prompt_id = '{0}_({1})_{2}.jpg'.format(
            image_path.stem,
            '{0:.0f}_{1:.0f}_{2:.0f}_{3:.0f}'.format(*bbox),
            '{0}_{1}'.format(*image_size if image_size else (0, 0))
        )
        image_prompt_path = target_dir / image_prompt_id

        if not image_prompt_path.is_file():
            image_prompt(
                image_path=image_path,
                bbox=bbox,
                origin_size=image_size,
                save_path=image_prompt_path
            )
        image_prompt_paths.append(image_prompt_path)

    return image_prompt_paths


def text_prompt(tokens, event_type, trigger, entity):
    sep_token = ['[SEP]']
    entity_token = ['$']

    tokens = tokens[:entity[1]] + entity_token + \
             tokens[entity[1]:entity[2]] + entity_token + \
             tokens[entity[2]:]

    tokens = tokens + sep_token + [trigger[0]] + [event_type]
    return tokens


# ACE, M2E2 Shared
class TextEvent(Dataset):
    def __init__(self, path, schema, tagger, max_length=200, keep_event=False):
        self.event_types = schema.events
        self.tagger = tagger
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

        self.max_length = max_length
        self.keep_event = keep_event

        self.sentence_ids = []
        self.token_lists = []
        self.triggers = []
        self.labels = []
        with open(path, 'r') as f:
            for i, line in enumerate(json.load(f)):
                sentence_id = line.get('sentence_id', i)
                token_list = line['words']
                triggers = [(
                    mention['event_type'],
                    mention['trigger']['start'],
                    mention['trigger']['end']
                ) for mention in line['golden-event-mentions']
                    if mention['event_type'] in self.event_types]
                labels = self.tagger.encode(
                    seq=[self.tagger.empty_tag] * len(token_list),
                    tag_offset=triggers)

                if self.keep_event and not triggers:
                    continue
                else:
                    self.sentence_ids.append(sentence_id)
                    self.token_lists.append(token_list)
                    self.triggers.append(triggers)
                    self.labels.append(labels)

    def post_process(self, predictions, refine=False):
        results = []
        if refine:
            predictions = refine_result(self.token_lists, self.labels, self.tagger.i2t_batch(predictions))
        for prediction in predictions:
            results.append(self.tagger.decode(prediction))
        return results

    def save_result(self, path, predictions):
        with open(path, 'w', encoding='utf-8') as f:
            save_result = {}
            for sentence_id, gold, prediction in zip(self.sentence_ids, self.triggers, predictions):
                save_result.setdefault(sentence_id, {})
                save_result[sentence_id]['golds'] = gold
                save_result[sentence_id]['predictions'] = prediction
            json.dump(save_result, f, indent=4)

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, item):
        token_ids = []
        token_id_mask = []
        sub_token_mask = []
        labels = []

        for token, label in zip(self.token_lists[item], self.tagger.t2i(self.labels[item])):
            sub_token_id = self.tokenizer(token, add_special_tokens=False)['input_ids']
            if len(sub_token_id) > 0:
                token_ids.extend(sub_token_id)
                token_id_mask.extend([1] * len(sub_token_id))
                sub_token_mask.extend([1] + [0] * (len(sub_token_id) - 1))
                if label % 2 == 1:
                    labels.extend([label] + [label + 1] * (len(sub_token_id) - 1))
                else:
                    labels.extend([label] * len(sub_token_id))
            else:
                assert 0

        # Special token
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_id_mask = [1] + token_id_mask + [1]
        sub_token_mask = [0] + sub_token_mask + [0]
        labels = [0] + labels + [0]

        padding = [0] * (self.max_length - len(token_ids))
        return (torch.LongTensor(token_ids + padding),
                torch.BoolTensor(token_id_mask + padding),
                torch.BoolTensor(sub_token_mask + padding),
                torch.LongTensor(labels + padding))


class TextArgument(Dataset):
    def __init__(self, path, schema, max_length=200, event_pred_path=None):
        with open(path, 'r') as f:
            self.raw_data = json.load(f)

        self.schema = schema
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.max_length = max_length

        # event pred
        self.event_prediction = load_json(event_pred_path) if event_pred_path else None

        self.sentence_ids = []
        self.token_lists = []
        self.event_types = []
        self.triggers = []
        self.entities = []
        self.labels = []

        for i, line in enumerate(self.raw_data):
            sentence_id = line.get('sentence_id', i)
            token_list = line['words']
            entity_golds = [(
                ' '.join(token_list[entity['start']:entity['end']]),
                entity['start'],
                entity['end']
            ) for entity in line['golden-entity-mentions']]

            # M2E2, Only for test
            if self.event_prediction:
                for event in self.event_prediction[sentence_id]:
                    for entity in entity_golds:
                        self.sentence_ids.append(sentence_id)
                        self.token_lists.append(token_list)
                        self.event_types.append(event[0])
                        self.triggers.append((
                            ' '.join(token_list[event[1]:event[2]]),
                            event[1],
                            event[2]
                        ))
                        self.entities.append(entity)
                        self.labels.append('None')
            # M2E2/ACE
            else:
                for event in line['golden-event-mentions']:
                    arguments = {(argument['start'], argument['end']): argument['role']
                                 for argument in event['arguments']}
                    for entity in entity_golds:
                        event_type = event['event_type']
                        if event_type not in self.schema.events:
                            continue
                        label = arguments.get((entity[1], entity[2]), 'None')
                        if label not in self.schema.e_roles:
                            continue

                        self.sentence_ids.append(sentence_id)
                        self.token_lists.append(token_list)
                        self.event_types.append(event_type)
                        self.triggers.append((
                            ' '.join(token_list[event['trigger']['start']:event['trigger']['end']]),
                            event['trigger']['start'],
                            event['trigger']['end']
                        ))
                        self.entities.append(entity)
                        self.labels.append(label)

        self.golds = {}
        for i, line in enumerate(self.raw_data):
            sentence_id = line.get('sentence_id', i)
            self.golds.setdefault(sentence_id, [])
            for mention in line['golden-event-mentions']:
                trigger = (
                    mention['event_type'],
                    mention['trigger']['start'],
                    mention['trigger']['end']
                )
                self.golds[sentence_id].extend([(
                    trigger,
                    argument['role'],
                    argument['start'],
                    argument['end']
                ) for argument in mention['arguments']])

    def post_process(self, predictions):
        results = {}
        for sentence_id, event_type, trigger, entity, prediction in zip(
                self.sentence_ids, self.event_types, self.triggers, self.entities, predictions):
            results.setdefault(sentence_id, [])
            trigger = (event_type, trigger[1], trigger[2])
            prediction = self.schema.e_roles[prediction]
            if prediction == 'None' or prediction not in self.schema.event2role[event_type]:
                continue
            results[sentence_id].append((trigger, prediction, entity[1], entity[2]))

        return {key: results.get(key, []) for key in self.golds.keys()}

    def save_result(self, path, predictions):
        with open(path, 'w', encoding='utf-8') as f:
            save_result = {}
            for sentence_id in self.golds.keys():
                save_result.setdefault(sentence_id, {})
                save_result[sentence_id]['golds'] = self.golds[sentence_id]
                save_result[sentence_id]['predictions'] = predictions[sentence_id]
            json.dump(save_result, f, indent=4)

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, item):
        tokens = text_prompt(
            self.token_lists[item],
            self.event_types[item],
            self.triggers[item],
            self.entities[item]
        )

        token_ids = []
        for token in tokens:
            token_ids.extend(self.tokenizer(token, add_special_tokens=False)['input_ids'])
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_id_mask = [1] * len(token_ids)

        label = self.schema.e_roles.index(self.labels[item])

        padding = [0] * (self.max_length - len(token_ids))
        return (torch.LongTensor(token_ids + padding),
                torch.BoolTensor(token_id_mask + padding),
                label)


class ImageEvent(Dataset):
    def __init__(self, path, image_dir, schema):
        self.image_dir = image_dir
        self.processor = AutoProcessor.from_pretrained(clip_path)

        self.verbs = schema.verbs
        self.events = schema.events
        self.verb2event = schema.verb2event

        self.image_ids = []
        self.labels = []
        with open(path, 'r') as f:
            for image, label in json.load(f).items():
                if label['verb'] in self.verbs:
                    self.image_ids.append(image)
                    self.labels.append([self.verb2event[label['verb']]])

    def post_process(self, predictions):
        results = []
        for prediction in predictions:
            prediction = [i for i in range(len(prediction)) if prediction[i] == 1]
            results.append([self.events[p] for p in prediction])
        return results

    def save_result(self, path, predictions):
        with open(path, 'w', encoding='utf-8') as f:
            save_result = {}
            for image_id, gold, prediction in zip(self.image_ids, self.labels, predictions):
                save_result.setdefault(image_id, {})
                save_result[image_id]['golds'] = gold
                save_result[image_id]['predictions'] = prediction
            json.dump(save_result, f, indent=4)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_feature = self.processor(
            images=Image.open(self.image_dir / self.image_ids[item]),
            return_tensors="pt"
        )['pixel_values'][0]

        label = [0] * len(self.events)
        for ll in self.labels[item]:
            label[self.events.index(ll)] = 1

        return image_feature, torch.FloatTensor(label)


class M2E2ImageEvent(Dataset):
    def __init__(self, path, image_dir, schema):
        self.image_dir = image_dir
        self.processor = AutoProcessor.from_pretrained(clip_path)

        self.events = schema.events

        with open(path, 'r') as f:
            data = json.load(f)
            self.image_ids = [f'{line}.jpg' for line in data.keys()]
            self.labels = [[line['event_type']] for line in data.values()]

    def post_process(self, predictions):
        results = []
        for prediction in predictions:
            prediction = [i for i in range(len(prediction)) if prediction[i] == 1]
            results.append([self.events[p] for p in prediction])
        return results

    def save_result(self, path, predictions):
        with open(path, 'w', encoding='utf-8') as f:
            save_result = {}
            for image_id, gold, prediction in zip(self.image_ids, self.labels, predictions):
                save_result.setdefault(image_id, {})
                save_result[image_id]['golds'] = gold
                save_result[image_id]['predictions'] = prediction
            json.dump(save_result, f, indent=4)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_feature = self.processor(
            images=Image.open(self.image_dir / self.image_ids[item]),
            return_tensors="pt"
        )['pixel_values'][0]
        # Not used
        label = [0] * len(self.events)

        return image_feature, torch.FloatTensor(label)


class ImageArgument(Dataset):
    def __init__(self, path, image_dir, schema, objects=None):
        self.image_dir = image_dir
        self.cache_dir = Path(f'{str(self.image_dir)}_cache')
        self.cache_dir.mkdir() if not self.cache_dir.exists() else None

        if objects:
            with open(objects, 'r') as f:
                self.objects = json.load(f)
        else:
            self.objects = None

        self.processor = AutoProcessor.from_pretrained(clip_path)

        self.verbs = schema.verbs
        self.v_roles = schema.v_roles
        self.e_roles = schema.e_roles
        self.mapping = schema.mapping

        self.image_paths = []
        self.bboxes_with_size = []
        self.labels = []
        with open(path, 'r') as f:
            for image, label in json.load(f).items():
                verb = label['verb']
                if verb not in self.verbs:
                    continue
                origin_size = (label['height'], label['width'])

                # From SWiG
                bboxes_checked = []
                for role, bbox in label['bb'].items():
                    if bbox[0] == -1:
                        continue
                    self.image_paths.append(self.image_dir / image)
                    self.bboxes_with_size.append([bbox, origin_size])
                    self.labels.append(self.mapping.get((verb, role), (verb, 'None'))[1])
                    bboxes_checked.append(bbox)

                # From Object detection
                if objects:
                    for bbox in self.objects[image]:
                        if bbox[1] < 0.25:
                            continue
                        for bbox_checked in bboxes_checked:
                            if iou(bbox[0], bbox_checked) > 0.5:
                                continue
                        self.image_paths.append(self.image_dir / image)
                        self.bboxes_with_size.append([bbox[0], origin_size])
                        self.labels.append('None')

        self.image_prompt_paths = build_prompt(
            self.image_paths, self.bboxes_with_size, self.cache_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_feature = self.processor(
            images=Image.open(self.image_prompt_paths[item]),
            return_tensors="pt"
        )['pixel_values'][0]
        label = self.e_roles.index(self.labels[item])

        return image_feature, label


class M2E2ImageArgument(Dataset):
    def __init__(self, path, image_dir, schema, objects=None):
        with open(path, 'r') as f:
            self.raw_data = json.load(f)
        self.image_dir = image_dir
        self.cache_dir = Path(f'{str(self.image_dir)}_cache')
        self.cache_dir.mkdir() if not self.cache_dir.exists() else None
        self.e_roles = schema.e_roles
        self.mapping = set(tuple(value) for value in schema.mapping.values())
        # Object detection or SWiG
        if objects:
            with open(objects, 'r') as f:
                self.objects = json.load(f)
        else:
            self.objects = None
        self.processor = AutoProcessor.from_pretrained(clip_path)

        self.image_paths = []
        self.bboxes_with_size = []
        for image, label in self.raw_data.items():
            image = f'{image}.jpg'
            if objects:
                for bbox in self.objects[image]:
                    if bbox[1] < 0.8:
                        continue
                    self.image_paths.append(self.image_dir / image)
                    self.bboxes_with_size.append((bbox[0], None))
            else:
                for role, bboxes in label['role'].items():
                    for bbox in bboxes:
                        self.image_paths.append(self.image_dir / image)
                        self.bboxes_with_size.append((bbox[1:], None))

        self.golds = {}
        self.event_golds = {}
        for key, value in self.raw_data.items():
            image_id = f'{key}.jpg'
            self.golds.setdefault(image_id, [])
            event_type = value['event_type']
            self.event_golds.setdefault(image_id, event_type)
            for role, bboxes in value['role'].items():
                for bbox in bboxes:
                    self.golds[image_id].append((event_type, role, bbox[1:]))

        self.image_prompt_paths = build_prompt(
            self.image_paths, self.bboxes_with_size, self.cache_dir)

    def post_process(self, predictions, event_prediction=None):
        if event_prediction:
            events = load_json(event_prediction)
        else:
            events = self.event_golds

        results = {}
        for image_path, prediction, bbox in zip(
                self.image_paths, predictions, self.bboxes_with_size):
            image = image_path.name
            results.setdefault(image, [])
            event = events[image]
            role = self.e_roles[prediction]
            if not event:
                continue
            if (event[0], role) not in self.mapping:
                continue
            results[image].append((event[0], role, bbox[0]))

        return {key: results.get(key, []) for key in self.golds.keys()}

    def save_result(self, path, predictions):
        with open(path, 'w', encoding='utf-8') as f:
            save_result = {}
            for image_id in self.golds.keys():
                save_result.setdefault(image_id, {})
                save_result[image_id]['golds'] = self.golds[image_id]
                save_result[image_id]['predictions'] = predictions[image_id]
            json.dump(save_result, f, indent=4)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_feature = self.processor(
            images=Image.open(self.image_prompt_paths[item]),
            return_tensors="pt"
        )['pixel_values'][0]
        # Not used
        label = 0

        return image_feature, label


class XMTL(nn.Module):
    def __init__(self, text_encoder, visual_encoder, tee_out_dim, tae_out_dim, vee_out_dim, vae_out_dim):
        super(XMTL, self).__init__()
        self.num_layers = 2
        self.hidden_size = 768

        # Modality-specific Input
        self.text_encoder = BertModel.from_pretrained(text_encoder)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.hidden_size)
        self.visual_encoder = CLIPVisionModel.from_pretrained(visual_encoder)
        self.visual_proj = nn.Linear(self.visual_encoder.config.hidden_size, self.hidden_size)

        # Task-shared Encoder
        self.multimodal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=8, batch_first=True)
        self.multimodal_encoder = nn.TransformerEncoder(
            encoder_layer=self.multimodal_encoder_layer, num_layers=self.num_layers)

        # Task-specific Output
        self.tee_fc = nn.Linear(self.hidden_size, tee_out_dim)
        self.tee_dropout = nn.Dropout(0.5)
        self.tee_crf = CRF(tee_out_dim, batch_first=True)

        self.tae_classifier = nn.Linear(self.hidden_size, tae_out_dim)
        self.tae_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.vee_classifier = nn.Linear(self.hidden_size, vee_out_dim)
        self.vee_sigmoid = nn.Sigmoid()
        self.vee_threshold = 0.5
        self.vee_criterion = nn.BCELoss(reduction='mean')

        self.vae_classifier = nn.Linear(self.hidden_size, vae_out_dim)
        self.vae_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch, task, is_test=False):
        if task == 'tee':
            token_ids, token_id_masks, sub_token_masks, labels = batch

            text_embed = self.text_proj(self.text_encoder(token_ids, token_id_masks).last_hidden_state)
            text_feature = self.multimodal_encoder(text_embed, src_key_padding_mask=token_id_masks)

            tee_feature = self.remove_sub_token(text_feature, sub_token_masks)
            token_id_masks = self.remove_sub_token(token_id_masks, sub_token_masks)
            tee_feature = self.tee_dropout(self.tee_fc(tee_feature))
            predictions = self.tee_crf.decode(tee_feature, token_id_masks)

            if is_test:
                return predictions
            else:
                labels = self.remove_sub_token(labels, sub_token_masks)
                loss = -self.tee_crf.forward(tee_feature, labels, token_id_masks, reduction='mean')
                return loss, predictions

        if task == 'tae':
            token_ids, token_id_masks, labels = batch

            text_embed = self.text_proj(self.text_encoder(token_ids, token_id_masks).last_hidden_state)
            text_feature = self.multimodal_encoder(text_embed, src_key_padding_mask=token_id_masks)

            tae_feature = F.avg_pool2d(text_feature, (text_feature.size(1), 1)).squeeze(1)
            predictions = self.tae_classifier(tae_feature)

            if is_test:
                return torch.argmax(predictions, dim=1)
            else:
                loss = self.tae_criterion(predictions, labels)
                return loss, torch.argmax(predictions, dim=1)

        elif task == 'vee':
            images, labels = batch
            image_embed = self.visual_proj(self.visual_encoder(images).last_hidden_state)
            image_feature = self.multimodal_encoder(image_embed)

            vee_feature = F.avg_pool2d(image_feature, (image_feature.size(1), 1)).squeeze(1)
            predictions = self.vee_sigmoid(self.vee_classifier(vee_feature))

            if is_test:
                predictions = predictions.cpu()
                return torch.where(predictions > self.vee_threshold, torch.tensor(1.), torch.tensor(0.))
            else:
                loss = self.vee_criterion(predictions, labels)
                predictions = predictions.cpu()
                return loss, torch.where(predictions > self.vee_threshold, torch.tensor(1.), torch.tensor(0.))

        if task == 'vae':
            images, labels = batch
            image_embed = self.visual_proj(self.visual_encoder(images).last_hidden_state)
            image_feature = self.multimodal_encoder(image_embed)

            vae_feature = F.avg_pool2d(image_feature, (image_feature.size(1), 1)).squeeze(1)
            predictions = self.vae_classifier(vae_feature)

            if is_test:
                return torch.argmax(predictions, dim=1)
            else:
                loss = self.vae_criterion(predictions, labels)
                return loss, torch.argmax(predictions, dim=1)

        else:
            raise NotImplementedError

    @staticmethod
    def remove_sub_token(embeds, sub_token_masks):
        length = embeds.shape[1]

        embeds_filter = []
        for embed, mask in zip(embeds, sub_token_masks):
            embed = embed[mask]
            pad_length = length - embed.shape[0]
            if len(embed.shape) == 2:
                pad = (0, 0, 0, pad_length)
            else:
                pad = (0, pad_length)
            embed = F.pad(embed, pad, 'constant', 0)
            embeds_filter.append(embed.unsqueeze(0))

        return torch.cat(embeds_filter, dim=0)


def train(train_data, task, model, optimizer, device):
    model.train()
    train_loss, train_n = 0.0, 0.0
    for batch in tqdm(train_data, desc=f'Train|{task}'):
        optimizer.zero_grad()
        batch = [b.to(device) for b in batch]
        loss, _ = model(batch, task)
        train_loss += loss.item()
        train_n += 1
        loss.backward()
        optimizer.step()
    train_loss = train_loss / train_n
    return model, train_loss

def alter_train(data, model, optimizer, device):
    tee_data, vee_data, tae_data, vae_data = data

    iterations = {
        'tee': DataLoader(tee_data, batch_size, shuffle=True),
        'vee': DataLoader(vee_data, batch_size, shuffle=True),
        'tae': DataLoader(tae_data, batch_size, shuffle=True),
        'vae': DataLoader(vae_data, batch_size, shuffle=True)
    }

    model.train()
    train_loss = {'tee': 0.0, 'vee': 0.0, 'tae': 0.0, 'vae': 0.0}

    tasks = ['tee', 'vae', 'tae', 'vee']
    for task in tasks:
        for batch in tqdm(iterations[task], desc=task):
            optimizer.zero_grad()
            batch = [b.to(device) for b in batch]
            loss, _ = model(batch, task)
            train_loss[task] += loss.item()
            loss.backward()
            optimizer.step()
        train_loss[task] = train_loss[task] / len(iterations[task])

    return model, train_loss


def valid(val_data, task, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(val_data, desc=f'Valid|{task}'):
            batch = [b.to(device) for b in batch]
            prediction = model(batch, task, is_test=True)
            predictions.extend(prediction)
    return predictions


def main(lr):
    logger.info(f'\nlr: {lr}')
    set_seed(seed)

    schema = M2E2Schema(m2e2_schema, add_none=False)
    schema_with_none = M2E2Schema(m2e2_schema, add_none=True)
    tagger = BIO(schema.events)

    # Data
    # Text Event
    ace_event_train = TextEvent(ace_train, schema, tagger)
    ace_event_dev = TextEvent(ace_dev, schema, tagger)
    ace_event_test = TextEvent(ace_test, schema, tagger)
    m2e2_tee_test = TextEvent(m2e2_text_test, schema, tagger, max_length=300)

    ace_event_train_iter = DataLoader(ace_event_train, batch_size, shuffle=True)
    ace_event_dev_iter = DataLoader(ace_event_dev, batch_size, shuffle=False)
    ace_event_test_iter = DataLoader(ace_event_test, batch_size, shuffle=False)
    m2e2_tee_test_iter = DataLoader(m2e2_tee_test, batch_size, shuffle=False)

    # Text Argument (The initiation of m2e2 is after TAE task.)
    ace_arg_train = TextArgument(ace_train, schema_with_none)
    ace_arg_dev = TextArgument(ace_dev, schema_with_none)
    ace_arg_test = TextArgument(ace_test, schema_with_none)

    ace_arg_train_iter = DataLoader(ace_arg_train, batch_size, shuffle=True)
    ace_arg_dev_iter = DataLoader(ace_arg_dev, batch_size, shuffle=False)
    ace_arg_test_iter = DataLoader(ace_arg_test, batch_size, shuffle=False)

    # Image Event
    imSitu_event_train = ImageEvent(imSitu_train, imSitu_image, schema)
    imSitu_event_dev = ImageEvent(imSitu_dev, imSitu_image, schema)
    imSitu_event_test = ImageEvent(imSitu_test, imSitu_image, schema)
    m2e2_vee_test = M2E2ImageEvent(m2e2_image_test, m2e2_image_path, schema)

    imSitu_event_train_iter = DataLoader(imSitu_event_train, batch_size, shuffle=True, num_workers=8)
    imSitu_event_dev_iter = DataLoader(imSitu_event_dev, batch_size, shuffle=False, num_workers=8)
    imSitu_event_test_iter = DataLoader(imSitu_event_test, batch_size, shuffle=False, num_workers=8)
    m2e2_vee_test_iter = DataLoader(m2e2_vee_test, batch_size, shuffle=False, num_workers=8)

    # Image Argument
    imSitu_arg_train = ImageArgument(swig_train, imSitu_image, schema_with_none)
    imSitu_arg_dev = ImageArgument(swig_dev, imSitu_image, schema_with_none)
    imSitu_arg_test = ImageArgument(swig_test, imSitu_image, schema_with_none)
    m2e2_vae_test = M2E2ImageArgument(m2e2_image_test, m2e2_image_path, schema_with_none, m2e2_object)

    imSitu_arg_train_iter = DataLoader(imSitu_arg_train, batch_size, shuffle=True, num_workers=8)
    imSitu_arg_dev_iter = DataLoader(imSitu_arg_dev, batch_size, shuffle=False, num_workers=8)
    imSitu_arg_test_iter = DataLoader(imSitu_arg_test, batch_size, shuffle=False, num_workers=8)
    m2e2_vae_test_iter = DataLoader(m2e2_vae_test, batch_size, shuffle=False, num_workers=8)

    # Model
    model = XMTL(
        text_encoder=bert_path,
        visual_encoder=clip_path,
        tee_out_dim=len(tagger.index2tag),
        tae_out_dim=len(schema_with_none.e_roles),
        vee_out_dim=len(schema.events),
        vae_out_dim=len(schema_with_none.e_roles)
    )
    model.to(device)

    # Optimizer
    # Frozen Backbone
    if lr[3]:
        for name, parameter in model.text_encoder.named_parameters():
            parameter.requires_grad = False
        for name, parameter in model.visual_encoder.named_parameters():
            parameter.requires_grad = False

    text_encoder_params = list(map(id, model.text_encoder.parameters()))
    visual_encoder_params = list(map(id, model.visual_encoder.parameters()))
    param_groups = text_encoder_params + visual_encoder_params
    other_params = filter(lambda x: id(x) not in param_groups, model.parameters())
    optimizer = AdamW([
        {'params': model.text_encoder.parameters(), 'lr': lr[0]},
        {'params': model.visual_encoder.parameters(), 'lr': lr[1]},
        {'params': other_params, 'lr': lr[2]}
    ])

    task = lr[4]
    best_dev_f1 = 0.0
    best_m2e2_f1 = 0.0
    for e in range(epoch):
        logger.info(f'\nEpoch: {e + 1}/{epoch}')
        best_dev_f1 = 0.0

        if task == 'tee':
            model, loss = train(ace_event_train_iter, task, model, optimizer, device)
            logger.info(f'loss：{loss:.4f}')

            # TEE
            ace_dev_pred = valid(ace_event_dev_iter, 'tee', model, device)
            ace_test_pred = valid(ace_event_test_iter, 'tee', model, device)
            m2e2_test_pred = valid(m2e2_tee_test_iter, 'tee', model, device)
            ace_dev_pred = ace_event_dev.post_process(ace_dev_pred)
            ace_test_pred = ace_event_test.post_process(ace_test_pred)
            m2e2_test_pred = m2e2_tee_test.post_process(m2e2_test_pred, refine=True)

            ace_dev_f1 = f_score(ace_dev_pred, ace_event_dev.triggers)
            ace_test_f1 = f_score(ace_test_pred, ace_event_test.triggers)
            tee_test_f1 = f_score(m2e2_test_pred, m2e2_tee_test.triggers)

            logger.info('ace_dev_f1: ({0:.4f},{1:.4f},{2:.4f})'.format(*ace_dev_f1))
            logger.info('ace_test_f1：({0:.4f},{1:.4f},{2:.4f})'.format(*ace_test_f1))
            logger.info('tee_test_f1: ({0:.4f},{1:.4f},{2:.4f})'.format(*tee_test_f1))

            if ace_dev_f1[2] > best_dev_f1:
                best_dev_f1 = ace_dev_f1[2]
                best_m2e2_f1 = tee_test_f1[2]
                torch.save(model.state_dict(), save_folder / 'tee_checkpoint_XMTL.pkl')
                m2e2_tee_test.save_result(text_event_pred_path, m2e2_test_pred)
                logger.info(f'model save!')

        elif task == 'vee':
            model, loss = train(imSitu_event_train_iter, task, model, optimizer, device)
            logger.info(f'loss：{loss:.4f}')

            # VEE
            imSitu_dev_pred = valid(imSitu_event_dev_iter, 'vee', model, device)
            imSitu_test_pred = valid(imSitu_event_test_iter, 'vee', model, device)
            m2e2_test_pred = valid(m2e2_vee_test_iter, 'vee', model, device)
            imSitu_dev_pred = imSitu_event_dev.post_process(imSitu_dev_pred)
            imSitu_test_pred = imSitu_event_test.post_process(imSitu_test_pred)
            m2e2_test_pred = m2e2_vee_test.post_process(m2e2_test_pred)

            imSitu_dev_f1 = f_score(imSitu_dev_pred, imSitu_event_dev.labels)
            imSitu_test_f1 = f_score(imSitu_test_pred, imSitu_event_test.labels)
            vee_test_f1 = f_score(m2e2_test_pred, m2e2_vee_test.labels)

            logger.info('imSitu_dev_f1: ({0:.4f},{1:.4f},{2:.4f})'.format(*imSitu_dev_f1))
            logger.info('imSitu_test_f1：({0:.4f},{1:.4f},{2:.4f})'.format(*imSitu_test_f1))
            logger.info('vee_test_f1: ({0:.4f},{1:.4f},{2:.4f})'.format(*vee_test_f1))

            if imSitu_dev_f1[2] > best_dev_f1:
                best_dev_f1 = imSitu_dev_f1[2]
                best_m2e2_f1 = vee_test_f1[2]
                torch.save(model.state_dict(), save_folder / 'vee_checkpoint_XMTL.pkl')
                m2e2_vee_test.save_result(image_event_pred_path, m2e2_test_pred)
                logger.info(f'model save!')

        elif task == 'tae':
            model, loss = train(ace_arg_train_iter, task, model, optimizer, device)
            logger.info(f'loss：{loss:.4f}')

            # TAE
            m2e2_tae_test = TextArgument(
                m2e2_text_test,
                schema_with_none,
                event_pred_path=text_event_pred_path,
                max_length=300
            )
            m2e2_tae_test_iter = DataLoader(m2e2_tae_test, batch_size, shuffle=False)
            ace_dev_pred = valid(ace_arg_dev_iter, 'tae', model, device)
            ace_test_pred = valid(ace_arg_test_iter, 'tae', model, device)
            m2e2_test_pred = valid(m2e2_tae_test_iter, 'tae', model, device)
            ace_dev_pred = [schema_with_none.e_roles[item] for item in ace_dev_pred]
            ace_test_pred = [schema_with_none.e_roles[item] for item in ace_test_pred]
            m2e2_test_pred = m2e2_tae_test.post_process(m2e2_test_pred)

            ace_dev_acc = acc_score(ace_dev_pred, ace_arg_dev.labels)
            ace_test_acc = acc_score(ace_test_pred, ace_arg_test.labels)
            tae_test_f1 = f_score(m2e2_test_pred, m2e2_tae_test.golds)

            logger.info('ace_dev_acc: {0:.4f}'.format(ace_dev_acc))
            logger.info('ace_test_acc：{0:.4f}'.format(ace_test_acc))
            logger.info('tae_test_f1: ({0:.4f},{1:.4f},{2:.4f})'.format(*tae_test_f1))

            if ace_dev_acc > best_dev_f1:
                best_dev_f1 = ace_dev_acc
                best_m2e2_f1 = tae_test_f1[2]
                torch.save(model.state_dict(), save_folder / 'tae_checkpoint_XMTL.pkl')
                m2e2_tae_test.save_result(text_arg_pred_path, m2e2_test_pred)
                logger.info(f'model save!')

        elif task == 'vae':
            model, loss = train(imSitu_arg_train_iter, task, model, optimizer, device)
            logger.info(f'loss：{loss:.4f}')

            # VAE
            imSitu_dev_pred = valid(imSitu_arg_dev_iter, 'vae', model, device)
            imSitu_test_pred = valid(imSitu_arg_test_iter, 'vae', model, device)
            m2e2_test_pred = valid(m2e2_vae_test_iter, 'vae', model, device)
            imSitu_dev_pred = [schema_with_none.e_roles[item] for item in imSitu_dev_pred]
            imSitu_test_pred = [schema_with_none.e_roles[item] for item in imSitu_test_pred]
            m2e2_test_pred = m2e2_vae_test.post_process(
                m2e2_test_pred, event_prediction=image_event_pred_path)

            imSitu_dev_acc = acc_score(imSitu_dev_pred, imSitu_arg_dev.labels)
            imSitu_test_acc = acc_score(imSitu_test_pred, imSitu_arg_test.labels)
            vae_test_f1 = f_score_iou(m2e2_test_pred, m2e2_vae_test.golds)

            logger.info('imSitu_dev_acc: {0:.4f}'.format(imSitu_dev_acc))
            logger.info('imSitu_test_acc：{0:.4f}'.format(imSitu_test_acc))
            logger.info('vae_test_f1: ({0:.4f},{1:.4f},{2:.4f})'.format(*vae_test_f1))

            if imSitu_dev_acc > best_dev_f1:
                best_dev_f1 = imSitu_dev_acc
                best_m2e2_f1 = vae_test_f1[2]
                torch.save(model.state_dict(), save_folder / 'vae_checkpoint_XMTL.pkl')
                m2e2_vae_test.save_result(image_arg_pred_path, m2e2_test_pred)
                logger.info(f'model save!')

    return best_m2e2_f1


if __name__ == '__main__':
    # Log config
    log_path.unlink() if log_path.exists() else None
    logger.remove()
    logger.add(sys.stdout, level='INFO', format='{message}')
    logger.add(log_path, level='INFO', format='{message}')

    result = []
    for lr in lr_list:
        result.append(main(lr))

    logger.info(result)
