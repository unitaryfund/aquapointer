FROM python:3

WORKDIR /aquapointer

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./aquapointer/analog/automated_analog_flow.py" ]
