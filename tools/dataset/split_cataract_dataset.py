import csv
import os
import random
import shutil
import matplotlib.pyplot as plt


def main():
    data_path = '/data/wangjinhong/data/cataract'
    output_path = 'data/cataract/output_stratified'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    val_split = 0.2
    data = {}
    for file_name in os.listdir(data_path):
        if file_name.startswith('C_all_data'):
            with open(os.path.join(data_path, file_name)) as f:
                data[file_name] = list(csv.reader(f))
    data['C_all_data_add_20211007_xz.csv'] = data['C_all_data_add_20211007_xz.csv'][1:]
    dataset_to_label = {k: {x[0]: x[1] for x in data[k]} for k in data}
    img_to_label = {img: {k: v[img] for k, v in dataset_to_label.items()} for img in dataset_to_label['C_all_data_add_2.csv']}
    match = {img: all([list(v.values())[0] == d for d in v.values()]) for img, v in img_to_label.items()}
    min_max_label = {img: [str(min([int(d) for d in v.values()])), str(max([int(d) for d in v.values()]))] for img, v in
                     img_to_label.items()}
    data = data['C_all_data_add_2.csv']
    for i in range(len(data)):
        min_label, max_label = min_max_label[data[i][0]]
        data[i].insert(2, min_label)
        data[i].insert(3, max_label)
    random.shuffle(data)

    labels = [{}, {}]
    for d in data:
        i = 0
        # i = 0 if d[0] in match and match[d[0]] else 1
        if int(d[1]) not in labels[i]:
            labels[i][int(d[1])] = [d]
        else:
            labels[i][int(d[1])].append(d)

    min_cls = min(set(labels[0].keys()) | set(labels[1].keys()))
    max_cls = max(set(labels[0].keys()) | set(labels[1].keys()))
    cls_indices = list(range(min_cls, max_cls + 1))

    matched_distribution = [0 if i not in labels[0] else len(labels[0][i]) for i in cls_indices]
    unmatched_distribution = [0 if i not in labels[1] else len(labels[1][i]) for i in cls_indices]
    all_distribution = [x + y for x, y in zip(matched_distribution, unmatched_distribution)]

    plt.bar(cls_indices, matched_distribution)
    plt.savefig(os.path.join(output_path, 'matched_distribution.png'))
    plt.cla()
    plt.bar(cls_indices, unmatched_distribution)
    plt.savefig(os.path.join(output_path, 'unmatched_distribution.png'))
    plt.cla()
    plt.bar(cls_indices, all_distribution)
    plt.savefig(os.path.join(output_path, 'all_distribution.png'))
    plt.cla()

    val_num = [round(x * val_split) for x in all_distribution]
    all_data, train_data, val_data = [], [], []
    for i, c in enumerate(range(min_cls, max_cls + 1)):
        all_data += labels[0].get(c, []) + labels[1].get(c, [])
        if val_num[i] < len(labels[0].get(c, [])):
            val_data += labels[0].get(c, [])[:val_num[i]]
            train_data += labels[0].get(c, [])[val_num[i]:] + labels[1].get(c, [])
        else:
            val_data += labels[0].get(c, []) + labels[1].get(c, [])[:val_num[i] - len(labels[0].get(c, []))]
            train_data += labels[1].get(c, [])[val_num[i] - len(labels[0].get(c, [])):]

    csv.writer(open(os.path.join(output_path, 'all.csv'), 'w')).writerows(all_data)
    csv.writer(open(os.path.join(output_path, 'train.csv'), 'w')).writerows(train_data)
    csv.writer(open(os.path.join(output_path, 'val.csv'), 'w')).writerows(val_data)


if __name__ == '__main__':
    main()
