FROM python

WORKDIR /usr/src

RUN apt update 
RUN apt install git -y
RUN git clone https://github.com/jpfleischer/chocolatechip.git
RUN pip install --upgrade pip

RUN cd chocolatechip; pip install -r requirements.txt

RUN --mount=type=secret,id=ACCESS_TOKEN \
    cat /run/secrets/ACCESS_TOKEN > /usr/src/access_token.txt &&
    
COPY statistics.py ./
#CMD ["python3","statistics.py"]
CMD ls

#remember to cd into this directory
#run with this :    docker build -t statsdocker .
#                   docker build --secret id=ACCESS_TOKEN,src=$PWD/super_secret.env -t statsdocker .

