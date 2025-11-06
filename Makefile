.All: help
.PHONY: splits ingest crawl help test test-model test-unit
.PHONY: venv install
.PHONY: train eval serve

help: 
	@echo "Makefile for the forecasting project"
	@echo ""
	@echo "Service run:"
	@echo " \tcrawl       - Run the crawl module"
	@echo " \tingest      - Run the ingest module"
	@echo " \tsplits      - Create data training/validation/test splits"
	@echo ""
	@echo "Environment setup:"
	@echo " \tvenv        - Create a virtual environment"
	@echo " \tinstall     - Install dependencies"
	@echo ""
	@echo "Training and evaluation:"
	@echo " \ttrain       - Train the model"
	@echo " \teval        - Evaluate the model"
	@echo " \ttest-model  - Comprehensive model testing with metrics and plots"
	@echo " \ttest-unit   - Run unit tests"
	@echo " \ttest        - Run all tests"
	@echo ""
	@echo " \thelp       - Show this help message"
splits:
	python -m src.make_splits --cfg configs/default.yaml --out_dir data/processed --format parquet --scaled

ingest:
	python -m src.ingest --crawl_dir data/crawl --out data/sample.csv --tz Asia/Bangkok

train:
	python -m src.train --cfg configs/default.yaml

eval:
	python -m src.evaluate --cfg configs/default.yaml --ckpt models/tft-best.ckpt

test-model:
	python -m src.water_forecast.test_model --cfg configs/default.yaml

test-unit:
	pytest tests/ -v

test: test-unit test-model

venv:
	python3 -m venv .venv

install:
	pip install -r requirements.txt

crawl:
	python -m src.crawl.main