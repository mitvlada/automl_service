FROM python:3.8-slim

WORKDIR /app

COPY ./requirements.txt ./

RUN apt-get update
RUN apt-get install libgomp1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001:5000

ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
