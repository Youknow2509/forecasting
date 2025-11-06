.All: help
.PHONNY: crawl help

help: 
	@echo "Makefile for the forecasting project"
	@echo ""
	@echo "Available targets:"
	@echo "  crawl       - Run the crawl module"
	@echo "  help        - Show this help message"

crawl:
	python -m src.crawl.main