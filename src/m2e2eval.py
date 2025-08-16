import re
import json
from pprint import pprint
from pathlib import Path
from itertools import product


# m2e2 dataset
m2e2_path = Path('../data/processed_data/m2e2_annotations')
m2e2_image = Path('../data/processed_data/m2e2_rawdata/image/image')


def load_score(path):
    score = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            score[(line[0], line[1])] = eval(line[2])

    return score


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        predictions = {key: value['predictions'] for key, value in data.items()}
        return predictions


def acc_score(predictions, golds):
    correct = 0
    total = len(predictions)
    for i in range(total):
        if predictions[i] == golds[i]:
            correct += 1
    return correct / total


def f_score(predictions, golds):
    if type(predictions) == dict:
        predictions = [predictions[key] for key in golds.keys()]
        golds = [golds[key] for key in golds.keys()]

    prediction_num = sum([len(prediction) for prediction in predictions])
    gold_num = sum([len(gold) for gold in golds])

    acc = 0
    for i in range(len(golds)):
        for p, g in product(predictions[i], golds[i]):
            if p == g:
                acc += 1

    p = acc / prediction_num if prediction_num else 0
    r = acc / gold_num if gold_num else 0
    f1 = 2 * p * r / (p + r) if p and r else 0

    return p, r, f1


def f_score_iou(predictions, golds):
    if type(predictions) == dict:
        predictions = [predictions[key] for key in golds.keys()]
        golds = [golds[key] for key in golds.keys()]

    prediction_num = sum([len(prediction) for prediction in predictions])
    gold_num = sum([len(gold) for gold in golds])

    acc = 0
    for i in range(len(golds)):
        for p, g in product(predictions[i], golds[i]):
            if p[0] == g[0] and p[1] == g[1] and iou(p[2], g[2]) > 0.5:
                acc += 1

    p = acc / prediction_num if prediction_num else 0
    r = acc / gold_num if gold_num else 0
    f1 = 2 * p * r / (p + r) if p and r else 0

    return p, r, f1


def iou(box_1, box_2):
    box_i = [
        max(box_1[0], box_2[0]),
        max(box_1[1], box_2[1]),
        min(box_1[2], box_2[2]),
        min(box_1[3], box_2[3])
    ]

    s_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    s_2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    s_i = max(0, box_i[2] - box_i[0]) * max(0, box_i[3] - box_i[1])
    s_o = s_1 + s_2 - s_i

    return s_i / s_o

# https://github.com/jianliu-ml/Multimedia-EE
def refine_result(words, tags_g, tags_p):
    result = [[word, tag_g, tag_p] for word, tag_g, tag_p in zip(words, tags_g, tags_p)]
    tags_p_filter = []

    word_dic = {}
    word_corret = {}
    word_corret_bi = {}

    for res in result:
        sens, gs, ps = res
        for idx, (w, g, p) in enumerate(zip(sens, gs, ps)):
            word_dic.setdefault(w, 0)
            word_dic[w] += 1

            word_corret.setdefault(w, {})
            word_corret[w].setdefault('t', 0)
            word_corret[w]['t'] += 1
            if p == g:
                word_corret[w].setdefault('c', 0)
                word_corret[w]['c'] += 1

            w = (sens[idx - 1], sens[idx])
            word_corret_bi.setdefault(w, {})
            word_corret_bi[w].setdefault('t', 0)
            word_corret_bi[w]['t'] += 1
            if p == g:
                word_corret_bi[w].setdefault('c', 0)
                word_corret_bi[w]['c'] += 1

    zzz = [(w, word_corret[w].get('c', 0) / word_corret[w]['t']) for w in word_corret]
    zzz = sorted(zzz, key=lambda x: x[1], reverse=False)[:100]
    wrong_set = set([x[0] for x in zzz])

    zzz = [(w, word_corret_bi[w].get('c', 0) / word_corret_bi[w]['t']) for w in word_corret_bi]
    zzz = sorted(zzz, key=lambda x: x[1], reverse=False)[:1000]
    wrong_set_bi = set([x[0] for x in zzz])

    for res in result:
        tag_p_filter = []
        sens, gs, ps = res

        count = 0
        for i in range(len(sens)):
            if sens[i] == '"':
                count += 1
            if count > 0 and count % 2 == 1:
                ps[i] = 'O'

        for idx, (w, g, p) in enumerate(zip(sens, gs, ps)):

            if w in wrong_set:
                p = 'O'

            if (sens[idx - 1], w) in wrong_set_bi:
                p = 'O'

            for i in range(max(0, idx - 10), min(len(sens), idx + 10)):
                if sens[i] == '“' or sens[i] == '”':
                    p = 'O'

            for i in range(max(0, idx - 9), min(len(sens), idx + 3)):
                if sens[i] == ':':
                    p = 'O'

            for i in range(max(0, idx - 3), min(len(sens), idx + 3)):
                if 'say' in sens[i] or 'avoid' in sens[i]:
                    p = 'O'

            if w == 'War' or w == 'deadly':
                p = 'O'

            if w == 'letter':
                p = 'O'

            if '-' in w:
                p = 'O'

            if w == 'talks' and not sens[idx - 1][0].isupper():
                p = 'O'

            if 'war-' in w:
                p = 'O'

            if 'email' in w:
                p = 'O'

            # if w == 'conflict' and p == 'B-Attack' and s == 'NN':
            #     p = 'O'

            # if w == 'war' and p == 'B-Attack' and s == 'NN':
            #     p = 'O'

            # if w == 'violence' and p == 'B-Attack' and s == 'NN':
            #     p = 'O'

            # if w == 'summit' and p == 'B-Meet' and s == 'NN':
            #     p = 'O'

            # if w == 'Summit' and p == 'B-Meet' and s == 'NNP':
            #     p = 'O'

            # if w == 'clashes' and p == 'B-Attack' and s == 'NNS':
            #     p = 'O'

            # if w == 'massacres' and s == 'NNS':
            #     p = 'O'

            # if w == 'suicide' and s == 'NN':
            #     p = 'O'

            # if w == 'shootings' and s == 'NNS':
            #     p = 'O'

            # if w == 'killings' and s == 'NNS':
            #     p = 'O'

            # if w == 'shelling' and s == 'VBG':
            #     p = 'O'

            # if w == 'murder' and s == 'NN':
            #     p = 'O'

            # if w == 'wars' and s == 'NNS':
            #     p = 'O'

            # if w == 'hostilities' and p == 'B-Attack' and s == 'NNS':
            #     p = 'O'

            # if w == 'die' and p == 'B-Die' and s == 'VB':
            #     p = 'O'

            # if w.lower().startswith('attack') and s.startswith('NN'):
            #     p = 'O'

            # if w.lower() == 'talks' and s.startswith('NN'):
            #     p = 'O'

            # if w.lower() == 'rally'  and s.startswith('NN'):
            #     p = 'O'

            tag_p_filter.append(p)
        tags_p_filter.append(tag_p_filter)

    return tags_p_filter


class M2E2Eval:
    def __init__(self, path):
        self.m2e2_path = Path(path)

        self.path = {
            'text_only_event': 'text_only_event.json',
            'text_multimedia_event': 'text_multimedia_event.json',
            'image_only_event': 'image_only_event.json',
            'image_multimedia_event': 'image_multimedia_event.json',
            'article_event': 'article_event.json',
            'image_event': 'image_event.json',
            'coref_event': 'crossmedia_coref.txt'
        }
        for key, value in self.path.items():
            self.path[key] = self.m2e2_path / value

    def eval(self, predictions, task):
        if task == 'text_event':
            return f_score(predictions, self.text_event())
        elif task == 'text_argument':
            return f_score(predictions, self.text_argument())
        elif task == 'image_event':
            return f_score(predictions, self.image_event())
        elif task == 'image_argument':
            return f_score_iou(predictions, self.image_argument())
        else:
            return 0, 0, 0

    def text_event(self):
        """
        Format: {sentence_id: [[event_type, trigger_start, trigger_end]*]}{N}
        """

        golds = {}
        with open(self.path['article_event'], 'r') as f:
            for line in json.load(f):
                sentence_id = line['sentence_id']
                golds.setdefault(sentence_id, [])
                for mention in line['golden-event-mentions']:
                    event = [
                        mention['event_type'],
                        mention['trigger']['start'],
                        mention['trigger']['end']
                    ]
                    golds[sentence_id].append(event)

        return golds

    def text_argument(self):
        """
        Format: {sentence_id: [[event, role_type, argument_start, argument_end]*]}{N}
        """

        golds = {}
        with open(self.path['article_event'], 'r') as f:
            for line in json.load(f):
                sentence_id = line['sentence_id']
                golds.setdefault(sentence_id, [])
                for mention in line['golden-event-mentions']:
                    event = [
                        mention['event_type'],
                        mention['trigger']['start'],
                        mention['trigger']['end']
                    ]
                    arguments = [[
                        event,
                        argument['role'],
                        argument['start'],
                        argument['end']
                    ] for argument in mention['arguments']]
                    golds[sentence_id].extend(arguments)

        return golds

    def image_event(self):
        """
        Format: {image_id: [event_type*]}{N}
        """

        golds = {}
        with open(self.path['image_event'], 'r') as f:
            for key, value in json.load(f).items():
                image_id = f'{key}.jpg'
                golds[image_id] = [value['event_type']]

        return golds

    def image_argument(self):
        """
        Format: {image_id: [[event_type, role_type, bbox]*]}{N}
        """

        golds = {}
        with open(self.path['image_event'], 'r') as f:
            for key, value in json.load(f).items():
                image_id = f'{key}.jpg'
                golds.setdefault(image_id, [])

                event_type = value['event_type']
                for role, bboxes in value['role'].items():
                    for bbox in bboxes:
                        golds[image_id].append((event_type, role, bbox[1:]))

        return golds

    def coref_event(self):
        """
        Format: [(sentence_id, image_id, event_type){N}]
        """

        with open(self.path['coref_event'], 'r') as f:
            golds = [tuple(line.strip().split('\t')) for line in f.readlines()]

        return golds

    def eval_coref(
            self,
            text_event_preds,
            image_event_preds,
            text_arg_preds,
            image_arg_preds,
            coref_scores
    ):

        # Filter ID
        with open(self.path['text_multimedia_event'], 'r') as f:
            sentence_ids = [line['sentence_id'] for line in json.load(f)]
            text_event_preds = {key: [e[0] for e in text_event_preds[key]] for key in sentence_ids}
        with open(self.path['image_multimedia_event'], 'r') as f:
            image_ids = [f'{key}.jpg' for key in json.load(f).keys()]
            image_event_preds = {key: image_event_preds[key] for key in image_ids}

        # Match
        coref_event_preds = []
        for pair, score in coref_scores.items():
            if pair[0] not in text_event_preds or pair[1] not in image_event_preds:
                continue
            if not image_event_preds[pair[1]] or not text_event_preds[pair[0]]:
                continue
            if score < 20:
                continue

            for image_event_pred in image_event_preds[pair[1]]:
                if image_event_pred in text_event_preds[pair[0]]:
                    coref_event_preds.append(pair + (image_event_pred,))

        # Eval
        coref_event_golds = self.coref_event()
        text_arg_golds = self.text_argument()
        image_arg_golds = self.image_argument()

        acc, t_acc, i_acc = 0, 0, 0
        tp_num, tg_num = 0, 0
        ip_num, ig_num = 0, 0

        for coref_event_pred in coref_event_preds:
            if coref_event_pred in coref_event_golds:
                sentence_id, image_id, event_type = coref_event_pred

                # Text event
                text_arg_pred = [tp for tp in text_arg_preds[sentence_id] if tp[0][0] == event_type]
                text_arg_gold = [tg for tg in text_arg_golds[sentence_id] if tg[0][0] == event_type]
                tp_num += len(text_arg_pred)
                tg_num += len(text_arg_gold)

                for tp, tg in product(text_arg_pred, text_arg_gold):
                    if tp == tg:
                        acc += 1
                        t_acc += 1

                # Image event
                image_arg_pred = [ip for ip in image_arg_preds[image_id] if ip[0] == event_type]
                image_arg_gold = [ig for ig in image_arg_golds[image_id] if ig[0] == event_type]
                ip_num += len(image_arg_pred)
                ig_num += len(image_arg_gold)

                for ip, ig in product(image_arg_pred, image_arg_gold):
                    if ip[0] == ig[0] and ip[1] == ig[1] and iou(ip[2], ig[2]) > 0.5:
                        i_acc += 1
                        acc += 1

        results = {'5 Multimodal Event': f_score([coref_event_preds], [coref_event_golds])}

        p = acc / (tp_num + ip_num)
        r = acc / (tg_num + ig_num)
        f1 = 2 * p * r / (p + r) if p and r else 0
        results['6 Multimodal Argument'] = (p, r, f1)

        p = t_acc / tp_num if tp_num else 0
        r = t_acc / tg_num if tg_num else 0
        f1 = 2 * p * r / (p + r) if p and r else 0
        results['7 Multimodal Argument (Text)'] = (p, r, f1)

        p = i_acc / ip_num if ip_num else 0
        r = i_acc / ig_num if ig_num else 0
        f1 = 2 * p * r / (p + r) if p and r else 0
        results['8 Multimodal Argument (Image)'] = (p, r, f1)

        return results


def eval_all(result_dir=Path('../outputs/1.Main_Result')):
    article_event_prediction = result_dir / 'predictions_m2e2_text_event.json'
    article_arg_prediction = result_dir / 'predictions_m2e2_text_arg.json'
    image_event_prediction = result_dir / 'predictions_m2e2_image_event.json'
    image_arg_prediction = result_dir / 'predictions_m2e2_image_arg.json'
    coref_score_path = Path('../outputs/m2e2-coref-CLIPScore/clip_score.txt')

    result = {}
    m2e2eval = M2E2Eval(m2e2_path)

    # Event
    article_event_predictions = load_json(article_event_prediction)
    image_event_predictions = load_json(image_event_prediction)

    # Argument
    article_arg_predictions = load_json(article_arg_prediction)
    image_arg_predictions = load_json(image_arg_prediction)

    # Conference
    coref_scores = load_score(coref_score_path)

    result['1 text_event'] = m2e2eval.eval(article_event_predictions, 'text_event')
    result['2 text_argument'] = m2e2eval.eval(article_arg_predictions, 'text_argument')
    result['3 image_event'] = m2e2eval.eval(image_event_predictions, 'image_event')
    result['4 image_argument'] = m2e2eval.eval(image_arg_predictions, 'image_argument')
    result.update(m2e2eval.eval_coref(
        article_event_predictions,
        image_event_predictions,
        article_arg_predictions,
        image_arg_predictions,
        coref_scores
    ))

    return result


if __name__ == '__main__':
    res = eval_all()
    for key, value in res.items():
        print(key, '{0:.3f},{1:.3f},{2:.3f}'.format(*value))

