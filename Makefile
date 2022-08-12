check:
	clear && echo "----- Pylint -----" && pylint src && echo "----- MyPy -----" && mypy src
