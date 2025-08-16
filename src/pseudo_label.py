import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, CLIPVisionModel, AutoTokenizer, AutoProcessor
from torchcrf import CRF

# Dataset Path
voa_path = Path('../data/processed_data/voa')
voa_image = voa_path / 'image'
voa_pair = voa_path / 'voa_img_dataset.json'
voa_text_entity = voa_path / 'voa_text_entity_StanfordNER.json'
voa_image_entity = voa_path / 'voa_image_entity_YOLOv8.json'

# Model Path
m2e2_schema = Path('../data/processed_data/m2e2_annotations/ace_sr_mapping.txt')
bert_path = Path('../weights/bert-base-uncased')
clip_path = Path('../weights/clip-vit-base-patch32')
tee_checkpoint = Path('../outputs/ace05-imsitu_event_XMTL_sep/tee_checkpoint_XMTL.pkl')
tae_checkpoint = Path('../outputs/ace05-imsitu_event_XMTL_sep/tae_checkpoint_XMTL.pkl')
vee_checkpoint = Path('../outputs/ace05-imsitu_event_XMTL_sep/vee_checkpoint_XMTL.pkl')
vae_checkpoint = Path('../outputs/ace05-imsitu_event_XMTL_sep/vae_checkpoint_XMTL.pkl')

# Outputs
save_folder = Path('../outputs/voa_pseudo_label/conf_0.80')
save_folder.mkdir(parents=True, exist_ok=True)
voa_tee_pred_rw_path = Path('../outputs/voa_pseudo_label/voa_tee_prediction_rw.json')
voa_vee_pred_rw_path = Path('../outputs/voa_pseudo_label/voa_vee_prediction_rw.json')
voa_tee_pred_path = save_folder / 'voa_tee_prediction.json'
voa_vee_pred_path = save_folder / 'voa_vee_prediction.json'
voa_tae_pred_path = save_folder / 'voa_tae_prediction.json'
voa_vae_pred_path = save_folder / 'voa_vae_prediction.json'
voa_text_pseudo_path = save_folder / 'voa_text_pseudo.json'
voa_image_pseudo_path = save_folder / 'voa_image_pseudo.json'
voa_coref_pseudo_path = save_folder / 'voa_coref_pseudo.txt'


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


def text_prompt(tokens, event_type, trigger, entity):
    sep_token = ['[SEP]']
    entity_token = ['$']

    tokens = tokens[:entity[1]] + entity_token + \
             tokens[entity[1]:entity[2]] + entity_token + \
             tokens[entity[2]:]

    tokens = tokens + sep_token + [trigger[0]] + [event_type]
    return tokens


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


class VOAPair:
    def __init__(self, path, image_dir, text_entity, image_entity):
        self.image_dir = image_dir

        with open(path, 'r') as f:
            self.raw_data = json.load(f)
        with open(text_entity, 'r') as f:
            self.text_entity = json.load(f)
        with open(image_entity, 'r') as f:
            self.image_entity = json.load(f)

        self.sentence_ids = []
        self.token_lists = []
        self.image_ids = []
        self.image_paths = []

        total = 0
        for doc_id, line in tqdm(self.raw_data.items(), desc='Load VOA'):
            for frag_id, pair in line.items():
                image_id = f'{doc_id}_{frag_id}.jpg'
                image_path = image_dir / image_id
                if not image_path.exists():
                    continue

                sentence_id = f'{doc_id}_{frag_id}'
                if sentence_id not in self.text_entity:
                    continue
                token_list = self.text_entity[sentence_id]['words']
                if not token_list:
                    continue

                self.sentence_ids.append(sentence_id)
                self.token_lists.append(token_list)
                self.image_ids.append(image_id)
                self.image_paths.append(image_path)
            total += 1


class VOATextEvent(Dataset):
    def __init__(self, voa, schema, tagger, max_length=200):
        self.sentence_ids = voa.sentence_ids
        self.token_lists = voa.token_lists

        self.event_types = schema.events
        self.tagger = tagger
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

    def post_process(self, predictions):
        results = []
        for prediction in predictions:
            results.append(self.tagger.decode(prediction))
        return results

    def save_result(self, path, predictions, conf_score):
        with open(path, 'w', encoding='utf-8') as f:
            save_result = {}
            for sentence_id, token_list, prediction, score in zip(
                    self.sentence_ids, self.token_lists, predictions, conf_score):
                save_result.setdefault(sentence_id, {})
                save_result[sentence_id]['predictions'] = prediction
                save_result[sentence_id]['conf_score'] = score
            json.dump(save_result, f, indent=4)

    def __len__(self):
        return len(self.sentence_ids)

    def __getitem__(self, item):
        token_ids = []
        token_id_mask = []
        sub_token_mask = []
        label = 0

        for token in self.token_lists[item]:
            sub_token_id = self.tokenizer(token, add_special_tokens=False)['input_ids']
            if len(sub_token_id) > 0:
                token_ids.extend(sub_token_id)
                token_id_mask.extend([1] * len(sub_token_id))
                sub_token_mask.extend([1] + [0] * (len(sub_token_id) - 1))
            else:
                print(token, self.token_lists[item], self.sentence_ids[item])

        # Special token
        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        token_id_mask = [1] + token_id_mask + [1]
        sub_token_mask = [0] + sub_token_mask + [0]

        padding = [0] * (self.max_length - len(token_ids))
        return (torch.LongTensor(token_ids + padding),
                torch.BoolTensor(token_id_mask + padding),
                torch.BoolTensor(sub_token_mask + padding),
                label)


class VOAImageEvent(Dataset):
    def __init__(self, voa, schema):
        self.image_ids = voa.image_ids
        self.image_paths = voa.image_paths

        self.event_types = schema.events
        self.processor = AutoProcessor.from_pretrained(clip_path)

    def post_process(self, predictions):
        results = []
        for prediction in predictions:
            prediction = [i for i in range(len(prediction)) if prediction[i] == 1]
            results.append([self.event_types[p] for p in prediction])
        return results

    def save_result(self, path, predictions, conf_score):
        with open(path, 'w', encoding='utf-8') as f:
            save_result = {}
            for image_id, prediction, score in zip(self.image_ids, predictions, conf_score):
                save_result.setdefault(image_id, {})
                save_result[image_id]['predictions'] = prediction
                save_result[image_id]['conf_score'] = score
            json.dump(save_result, f, indent=4)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_feature = self.processor(
            images=Image.open(self.image_paths[item]),
            return_tensors="pt"
        )['pixel_values'][0]
        label = 0

        return image_feature, label


class VOATextArgument(Dataset):
    def __init__(self, voa, tee_pred, schema, max_length=200):
        with open(tee_pred, 'r') as f:
            self.tee_pred = json.load(f)
        self.text_entity = voa.text_entity

        self.e_roles = schema.e_roles
        self.event2role = schema.event2role
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

        self.sentence_ids = []
        self.token_lists = []
        self.event_types = []
        self.triggers = []
        self.entities = []
        for sentence_id, predictions in self.tee_pred.items():
            token_list = self.text_entity[sentence_id]['words']
            for event in predictions['predictions']:
                for entity in self.text_entity[sentence_id]['golden-entity-mentions']:
                    self.sentence_ids.append(sentence_id)
                    self.token_lists.append(token_list)
                    self.event_types.append(event[0])
                    self.triggers.append((
                        ' '.join(token_list[event[1]:event[2]]),
                        event[1],
                        event[2]
                    ))
                    self.entities.append((
                        ' '.join(token_list[entity['start']:entity['end']]),
                        entity['start'],
                        entity['end']
                    ))

    def post_process(self, predictions, conf_score):
        results = {}
        for sentence_id, event_type, trigger, entity, prediction, score in zip(
                self.sentence_ids, self.event_types, self.triggers, self.entities, predictions, conf_score):
            results.setdefault(sentence_id, [])
            trigger = (event_type, trigger[1], trigger[2])
            prediction = self.e_roles[prediction]
            if prediction not in self.event2role[event_type] and prediction != 'None':
                continue
            results[sentence_id].append((trigger, prediction, entity[1], entity[2], score))

        return {key: results.get(key, []) for key in self.tee_pred.keys()}

    def save_result(self, path, predictions):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=4)

    def __len__(self):
        return len(self.sentence_ids)

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

        label = 0
        padding = [0] * (self.max_length - len(token_ids))
        return (torch.LongTensor(token_ids + padding),
                torch.BoolTensor(token_id_mask + padding),
                label)


class VOAImageArgument(Dataset):
    def __init__(self, voa, vee_pred, schema):
        with open(vee_pred, 'r') as f:
            self.vee_pred = json.load(f)
        self.image_entity = voa.image_entity

        self.e_roles = schema.e_roles
        self.mapping = set(tuple(value) for value in schema.mapping.values())
        self.processor = AutoProcessor.from_pretrained(clip_path)

        # For Image Prompt
        self.image_dir = voa.image_dir
        self.cache_dir = Path(f'{str(self.image_dir)}_cache')
        self.cache_dir.mkdir() if not self.cache_dir.exists() else None

        self.image_paths = []
        self.bboxes_with_size = []
        for image_id in self.vee_pred.keys():
            for bbox in self.image_entity[image_id]:
                if bbox[1] < 0.8:
                    continue
                self.image_paths.append(self.image_dir / image_id)
                self.bboxes_with_size.append((bbox[0], None))

        self.image_prompt_paths = build_prompt(
            self.image_paths, self.bboxes_with_size, self.cache_dir)

    def post_process(self, predictions, conf_score):
        results = {}
        for image_path, prediction, bbox, score in zip(
                self.image_paths, predictions, self.bboxes_with_size, conf_score):
            image_id = image_path.name
            results.setdefault(image_id, [])
            event = self.vee_pred[image_id]['predictions']
            role = self.e_roles[prediction]
            if not event:
                continue
            if (event[0], role) not in self.mapping and role != 'None':
                continue
            results[image_id].append((event[0], role, bbox[0], score))

        return {key: results.get(key, []) for key in self.vee_pred.keys()}

    def save_result(self, path, predictions):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=4)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_feature = self.processor(
            images=Image.open(self.image_prompt_paths[item]),
            return_tensors="pt"
        )['pixel_values'][0]
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

        # Teacher Loss
        self.coff = 0.1
        self.mse = torch.nn.MSELoss()

    def get_text_feature(self, token_id, token_id_mask):
        text_embed = self.text_proj(self.text_encoder(token_id, token_id_mask).last_hidden_state)
        text_feature = self.multimodal_encoder(text_embed, src_key_padding_mask=token_id_mask)
        return text_feature

    def get_image_feature(self, image):
        image_embed = self.visual_proj(self.visual_encoder(image).last_hidden_state)
        image_feature = self.multimodal_encoder(image_embed)
        return image_feature

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

    def task_tee(self, batch):
        token_ids, token_id_masks, sub_token_masks, label = batch
        text_feature = self.get_text_feature(token_ids, token_id_masks)

        tee_feature = self.remove_sub_token(text_feature, sub_token_masks)
        token_id_masks = self.remove_sub_token(token_id_masks, sub_token_masks)
        tee_feature = self.tee_dropout(self.tee_fc(tee_feature))
        predictions = self.tee_crf.decode(tee_feature, token_id_masks)

        tags = []
        for prediction in predictions:
            padding = [0] * (token_id_masks.shape[1] - len(prediction))
            tags.append(prediction + padding)
        tags = torch.LongTensor(tags).to(tee_feature.device)
        loss = self.tee_crf.forward(tee_feature, tags, token_id_masks, reduction='none')
        conf_score = torch.exp(loss).tolist()

        return predictions, conf_score

    def task_vee(self, batch):
        images, label = batch
        image_feature = self.get_image_feature(images)

        vee_feature = F.avg_pool2d(image_feature, (image_feature.size(1), 1)).squeeze(1)
        probs = self.vee_sigmoid(self.vee_classifier(vee_feature)).cpu()
        predictions = torch.where(probs > self.vee_threshold, torch.tensor(1.), torch.tensor(0.))
        predictions = predictions.type(torch.bool)

        conf_score = []
        for prob, prediction in zip(probs, predictions):
            conf_score.append(prob[prediction].tolist())

        return predictions, conf_score

    def task_tae(self, batch):
        token_ids, token_id_masks, labels = batch
        text_feature = self.get_text_feature(token_ids, token_id_masks)

        tae_feature = F.avg_pool2d(text_feature, (text_feature.size(1), 1)).squeeze(1)
        probs = F.softmax(self.tae_classifier(tae_feature), dim=1)
        predictions = torch.argmax(probs, dim=1)

        conf_score = []
        for prob, prediction in zip(probs, predictions):
            conf_score.append(prob[prediction].tolist())

        return predictions, conf_score

    def task_vae(self, batch):
        images, labels = batch
        image_feature = self.get_image_feature(images)

        vae_feature = F.avg_pool2d(image_feature, (image_feature.size(1), 1)).squeeze(1)
        probs = F.softmax(self.vae_classifier(vae_feature), dim=1)
        predictions = torch.argmax(probs, dim=1)

        conf_score = []
        for prob, prediction in zip(probs, predictions):
            conf_score.append(prob[prediction].tolist())

        return predictions, conf_score

    def forward(self, batch, task):
        if task == 'tee':
            return self.task_tee(batch)
        elif task == 'vee':
            return self.task_vee(batch)
        elif task == 'tae':
            return self.task_tae(batch)
        elif task == 'vae':
            return self.task_vae(batch)
        else:
            raise NotImplementedError


def valid(val_data, task, model, device):
    model.eval()
    predictions = []
    conf_scores = []
    with torch.no_grad():
        for batch in tqdm(val_data, desc=f'Valid|{task}'):
            batch = [b.to(device) for b in batch]
            prediction, conf_score = model(batch, task)
            predictions.extend(prediction)
            conf_scores.extend(conf_score)
    return predictions, conf_scores


def ee():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    schema = M2E2Schema(m2e2_schema, add_none=False)
    tagger = BIO(schema.events)

    # Load Data
    voa = VOAPair(voa_pair, voa_image, voa_text_entity, voa_image_entity)
    voa_text_event = VOATextEvent(voa, schema, tagger)
    voa_image_event = VOAImageEvent(voa, schema)
    voa_text_event_iter = DataLoader(voa_text_event, 64, shuffle=False)
    voa_image_event_iter = DataLoader(voa_image_event, 64, shuffle=False, num_workers=4)

    # Load Model
    model = XMTL(
        text_encoder=bert_path,
        visual_encoder=clip_path,
        tee_out_dim=len(tagger.index2tag),
        tae_out_dim=len(schema.e_roles) + 1,
        vee_out_dim=len(schema.events),
        vae_out_dim=len(schema.e_roles) + 1
    )

    # VOA Text Event
    state_dict = torch.load(tee_checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    print('Load TEE Model Done!')

    voa_text_event_pred, conf_score = valid(voa_text_event_iter, 'tee', model, device)
    voa_text_event_pred = voa_text_event.post_process(voa_text_event_pred)
    voa_text_event.save_result(voa_tee_pred_rw_path, voa_text_event_pred, conf_score)

    # VOA Image Event
    state_dict = torch.load(vee_checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    print('Load VEE Model Done!')

    voa_image_event_pred, conf_score = valid(voa_image_event_iter, 'vee', model, device)
    voa_image_event_pred = voa_image_event.post_process(voa_image_event_pred)
    voa_image_event.save_result(voa_vee_pred_rw_path, voa_image_event_pred, conf_score)


def coref(threshold=0.8):
    with open(voa_tee_pred_rw_path, 'r') as f:
        voa_tee_pred = json.load(f)
    with open(voa_vee_pred_rw_path, 'r') as f:
        voa_vee_pred = json.load(f)

    pair_event_pseudo = []
    for key in voa_tee_pred.keys():
        # Conf
        if voa_tee_pred[key]['conf_score'] < threshold:
            continue

        # Coref
        text_events = [event[0] for event in voa_tee_pred[key]['predictions']]
        image_events = voa_vee_pred[f'{key}.jpg']['predictions']
        for image_event, conf in zip(image_events, voa_vee_pred[f'{key}.jpg']['conf_score']):
            if conf < threshold:
                continue
            if image_event in text_events:
                pair_event_pseudo.append([key, f'{key}.jpg', image_event])

    voa_tee_pred_filter, voa_vee_pred_filter = {}, {}
    for sentence_id, image_id, _ in pair_event_pseudo:
        voa_tee_pred_filter.setdefault(sentence_id, voa_tee_pred[sentence_id])
        voa_vee_pred_filter.setdefault(image_id, voa_vee_pred[image_id])

    with open(voa_coref_pseudo_path, 'w') as f:
        for line in pair_event_pseudo:
            f.writelines('{0} {1} {2}'.format(*line))
            f.writelines('\n')
    with open(voa_tee_pred_path, 'w') as f:
        json.dump(voa_tee_pred_filter, f, indent=4)
    with open(voa_vee_pred_path, 'w') as f:
        json.dump(voa_vee_pred_filter, f, indent=4)


def ae():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    schema = M2E2Schema(m2e2_schema, add_none=False)
    schema_with_none = M2E2Schema(m2e2_schema, add_none=True)
    tagger = BIO(schema.events)

    # Load Data
    voa = VOAPair(voa_pair, voa_image, voa_text_entity, voa_image_entity)
    voa_text_arg = VOATextArgument(voa, voa_tee_pred_path, schema_with_none)
    voa_image_arg = VOAImageArgument(voa, voa_vee_pred_path, schema_with_none)
    voa_text_arg_iter = DataLoader(voa_text_arg, 64, shuffle=False)
    voa_image_arg_iter = DataLoader(voa_image_arg, 64, shuffle=False, num_workers=4)

    # Load Model
    model = XMTL(
        text_encoder=bert_path,
        visual_encoder=clip_path,
        tee_out_dim=len(tagger.index2tag),
        tae_out_dim=len(schema_with_none.e_roles),
        vee_out_dim=len(schema.events),
        vae_out_dim=len(schema_with_none.e_roles)
    )

    # VOA Text Argument
    state_dict = torch.load(tae_checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    print('Load TAE Model Done!')

    voa_text_arg_pred, conf_score = valid(voa_text_arg_iter, 'tae', model, device)
    voa_text_arg_pred = voa_text_arg.post_process(voa_text_arg_pred, conf_score)
    voa_text_arg.save_result(voa_tae_pred_path, voa_text_arg_pred)

    # VOA Image Argument
    state_dict = torch.load(vae_checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    print('Load VAE Model Done!')

    voa_image_arg_pred, conf_score = valid(voa_image_arg_iter, 'vae', model, device)
    voa_image_arg_pred = voa_image_arg.post_process(voa_image_arg_pred, conf_score)
    voa_image_arg.save_result(voa_vae_pred_path, voa_image_arg_pred)


def combine(threshold=0.8):
    with open(voa_text_entity, 'r') as f:
        text_entity = json.load(f)
    with open(voa_tee_pred_path, 'r') as f:
        tee_pred = json.load(f)
    with open(voa_vee_pred_path, 'r') as f:
        vee_pred = json.load(f)
    with open(voa_tae_pred_path, 'r') as f:
        tae_pred = json.load(f)
    with open(voa_vae_pred_path, 'r') as f:
        vae_pred = json.load(f)

    voa_text_pseudo = []
    for sentence_id in tae_pred.keys():
        token_list = text_entity[sentence_id]['words']
        entity = text_entity[sentence_id]['golden-entity-mentions']

        triggers = [tuple(trigger) for trigger in tee_pred[sentence_id]['predictions']]
        arguments = {trigger: [] for trigger in triggers}
        for argument in tae_pred[sentence_id]:
            if argument[-1] < threshold:
                continue
            if argument[1] == 'None':
                continue

            arguments[tuple(argument[0])].append({
                'role': argument[1],
                'text': token_list[argument[2]:argument[3]],
                'start': argument[2],
                'end': argument[3]
            })

        voa_text_pseudo.append({
            'sentence_id': sentence_id,
            'words': token_list,
            'golden-entity-mentions': entity,
            'golden-event-mentions': [{
                'trigger': {
                    'start': trigger[1],
                    'end': trigger[2],
                    'text': ' '.join(token_list[trigger[1]:trigger[2]])
                },
                'arguments': arguments[trigger],
                'event_type': trigger[0]
            } for trigger in triggers]
        })

    voa_image_pseudo = {}
    for image_id in vee_pred:
        def max_conf_pred(x, y):
            return x[y.index(max(y))]

        event_type = max_conf_pred(vee_pred[image_id]['predictions'], vee_pred[image_id]['conf_score'])

        arguments = {}
        for argument in vae_pred[image_id]:
            if argument[-1] < threshold:
                continue

            bbox = [1] + [int(x) for x in argument[2]]
            arguments.setdefault(argument[1], [])
            arguments[argument[1]].append(bbox)

        voa_image_pseudo[image_id[:-4]] = {
            'role': arguments,
            'event_type': event_type
        }

    with open(voa_text_pseudo_path, 'w') as f:
        json.dump(voa_text_pseudo, f, indent=4)
    with open(voa_image_pseudo_path, 'w') as f:
        json.dump(voa_image_pseudo, f, indent=4)


if __name__ == '__main__':
    ee()
    coref(0.80)
    ae()
    combine(0.80)
