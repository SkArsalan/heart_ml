FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001
#CMD ["python", "app.py"]

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]