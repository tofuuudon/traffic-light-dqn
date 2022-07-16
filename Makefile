run:
	python src/main.py
lint:
	pylint src
type:
	mypy src
check:
	clear && echo "----- Pylint -----" && pylint src && echo "----- MyPy -----" && mypy src
