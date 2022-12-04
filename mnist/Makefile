reqs.lock:
	pip-compile reqs.in -o reqs.lock

venv: reqs.lock
	virtualenv venv

update: venv
	. venv/bin/activate && pip install -r reqs.lock