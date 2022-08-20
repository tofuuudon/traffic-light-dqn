check:
	clear && echo "----- Pylint -----" && pylint src && echo "----- MyPy -----" && mypy src
generate-test:
	python3 src/main.py --episodes 50 --log --save-models;
	python3 src/main.py --episodes 50 --model-variant 1 --log --save-models;
	python3 src/main.py --episodes 50 --model-variant 2 --log --save-models;
	python3 src/main.py --episodes 50 --no-ai --log
