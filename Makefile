SHELL := /bin/sh

VENV := .venv
PY   := $(VENV)/bin/python

.DEFAULT_GOAL := default

.PHONY: default venv install clean

default: install

venv:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"

install: venv
	@$(PY) -m ensurepip --upgrade || true
	@$(PY) -m pip install --upgrade pip
	@$(PY) -m pip install -r requirements.txt
# usar `python -m pip` garante que instala na venv, não no global [web:40][web:81]

clean:
	@rm -rf "$(VENV)"
