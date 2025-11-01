import json 
import os

with open("data/annotation.json", "r") as f:
	annotations = json.load(f)

# Training
train_images_dir = "data/train/images"
train_labels_dir = "data/train/labels"
train_images = set(os.listdir(train_images_dir))

# Validation
val_images_dir = "data/val/images"
val_labels_dir = "data/val/labels"
val_images = set(os.listdir(val_images_dir))

class_map = {"mouse": 0, "keyboard": 1}

for item in annotations:
	image_name = item["image"]
	labels = item["label"] #list objects in image
	lines = [] # indiv. lines of YOLO file

	for obj in labels: #processing each object in the image
		class_name = obj["rectanglelabels"][0]
		if class_name not in class_map:
			print("Unkown class, skipped.")
			continue
		class_id = class_map[class_name]
		# YOLO formatting 
		x = obj["x"]
		y = obj["y"]
		width = obj["width"]
		height = obj["height"]

		x_center = (x + width/2) / 100
		y_center = (y + height/2) / 100
		width /= 100
		height /= 100

		lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
	# determine saving location of the label
	if image_name in train_images:
		label_path = os.path.join(train_labels_dir, os.path.splitext(image_name)[0] + ".txt")
	elif image_name in val_images:
		label_path = os.path.join(val_labels_dir, os.path.splitext(image_name)[0] + ".txt")
	else:
		print(f"{image_name} was not found, skipped.")
		continue
	# write label file
	with open(label_path, "w") as f:
		f.write("\n".join(lines))

print("Done creating labels")