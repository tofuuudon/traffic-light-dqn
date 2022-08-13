check:
	clear && echo "----- Pylint -----" && pylint src && echo "----- MyPy -----" && mypy src
generate-test:
	python3 src/main.py --episodes 20 --log;
	python3 src/main.py --episodes 20 --model-variant 1 --log;
	python3 src/main.py --episodes 20 --model-variant 2 --log;
