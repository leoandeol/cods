wget https://raw.githubusercontent.com/open-mmlab/mmdetection/main/tools/misc/download_dataset.py
python ./download_dataset.py --dataset-name coco2017
cd data/coco/ && unzip annotations_trainval2017.zip && unzip test2017.zip && unzip val2017.zip
