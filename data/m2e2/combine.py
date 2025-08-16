import json


def combine(file_1, file_2):
    """
    Combine json file
    """

    with open(file_1, 'r') as f_1, open(file_2, 'r') as f_2:
        data_1 = json.load(f_1)
        data_2 = json.load(f_2)

        # image event
        if isinstance(data_1, dict) and isinstance(data_2, dict):
            print(len(data_1))
            print(len(data_2))
            data_1.update(data_2)
            result = data_1
            print(len(result))

        # text event
        elif isinstance(data_1, list) and isinstance(data_2, list):
            print(len(data_1))
            print(len(data_2))
            num = 0
            data_1_id = [line['sentence_id'] for line in data_1]
            for line in data_2:
                if line['sentence_id'] not in data_1_id:
                    data_1.append(line)
                else:
                    num += 1
                    target_index = data_1_id.index(line['sentence_id'])
                    data_1[target_index]['golden-event-mentions'].extend(line['golden-event-mentions'])
            result = data_1
            print(num)
            print(len(result))

        else:
            raise 'Unsupported json format!'

    return result


if __name__ == '__main__':
    # Generate article_event.json
    with open('../../processed_data/m2e2_annotations/article_event.json', 'w') as f:
        combine_data = combine(
            file_1='../../processed_data/m2e2_annotations/text_only_event.json',
            file_2='../../processed_data/m2e2_annotations/text_multimedia_event.json'
        )
        json.dump(combine_data, f, indent=4)
    print('Generate article_event.json done!')

    # Generate image_event.json
    with open('../../processed_data/m2e2_annotations/image_event.json', 'w') as f:
        combine_data = combine(
            file_1='../../processed_data/m2e2_annotations/image_only_event.json',
            file_2='../../processed_data/m2e2_annotations/image_multimedia_event.json'
        )
        json.dump(combine_data, f, indent=4)
    print('Generate image_event.json done!')
