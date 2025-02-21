.PHONY: env
env:
	conda create -n calibr python=3.8.*

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: kernel
kernel:
	python -m ipykernel install --user --name=calibr