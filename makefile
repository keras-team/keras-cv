.PHONY: format clean

clean:
	rm -rf keras_cv.egg-info/
	rm -rf keras_cv/**/__pycache__
	rm -rf keras_cv/__pycache__
format:
	isort --sl .
	black .
