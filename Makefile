.PHONY: venv

VAL=http://images.cocodataset.org/zips/val2017.zip
ANNOTATIONS=http://images.cocodataset.org/annotations/annotations_trainval2017.zip

venv:
	$(CODS_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install --editable .
	

coco:
	$(info Downloading COCO dataset to: $(COCO_DIR))
	wget $(VAL) -P $(COCO_DIR)
	wget $(ANNOTATIONS) -P $(COCO_DIR)
	unzip val2017.zip
	rm val2017.zip
	unzip annotations_trainval2017.zip
	rm annotations_trainval2017.zip

dotenv:
	@echo "PROJECT_PATH=$$PWD" > .env
	@echo "COCO_PATH=$$COCO_DIR" >> .env

install:
	@make project_path
	@make venv
	@make coco
