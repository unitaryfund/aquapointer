FROM python:3

WORKDIR /usr/src/aquapointer

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

CMD [ "python", "./aquapointer/analog/automated_flow.py" ]
