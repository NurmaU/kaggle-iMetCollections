FROM python:3.6
WORKDIR /app
COPY requirements.txt /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
COPY . /app
ADD . /app
#CMD ["bash", "train.sh"]
CMD ["python", "show.py"]