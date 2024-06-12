FROM python:3

WORKDIR /root

# check basic installed packages
RUN apt-get --version
RUN git --version
RUN bash --version

RUN apt-get update
RUN apt-get install -y vim


COPY ./aquapointer /root/aquapointer 

WORKDIR /root/aquapointer

# Install packages

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "analog/automated_analog_flow.py" ]
