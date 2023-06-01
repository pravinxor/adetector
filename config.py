import csv

sample_rate = 32000

labels_csv_path = './class_labels_indices.csv'

# Load label
with open(labels_csv_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []  # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label: i for i, label in enumerate(labels)}
ix_to_lb = {i: label for i, label in enumerate(labels)}

id_to_ix = {id: i for i, id in enumerate(ids)}
ix_to_id = {i: id for i, id in enumerate(ids)}
