# remove old env (optional but recommended)
rmdir /s /q venv

# create new env with Python 3.12
python3.12 -m venv venv
venv\Scripts\activate

# upgrade pip
python -m pip install --upgrade pip

# install deps
RUN pip install --only-binary=:all: -r requirements.txt
