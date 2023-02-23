# start by pulling the python image
FROM python:3.9-alpine

RUN mkdir /myapp

COPY . /myapp

WORKDIR /myapp

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]