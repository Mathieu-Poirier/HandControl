FROM python:3.10

WORKDIR /CNNHOST

COPY . /CNNHOST

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r server_requirements.txt

EXPOSE 5000

ENV FLASK_APP=server.py

CMD ["flask", "run", "--host=0.0.0.0"]