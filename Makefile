check:
	clear && echo "----- Pylint -----" && pylint src && echo "----- MyPy -----" && mypy src
generate-test:
	python3 src/main.py --episodes 100 --max-step 100 --log;
	python3 src/main.py --episodes 100 --max-step 100 --model-variant 1 --log;
	python3 src/main.py --episodes 100 --max-step 100 --model-variant 2 --log;
