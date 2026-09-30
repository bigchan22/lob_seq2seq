.PHONY: test compile
test:
	CUDA_VISIBLE_DEVICES="" pytest -q
compile:
	python -m compileall lob_forecasting src
