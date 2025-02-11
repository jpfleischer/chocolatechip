FROM python:3.12

WORKDIR /usr/src

RUN apt update 
RUN pip install --upgrade pip

COPY statistics.py ./

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#cloning and such (irrelevant with volume mount)
#   RUN apt install git -y
#   RUN git clone https://github.com/jpfleischer/chocolatechip.git

#fixing some errors
RUN apt install nano
RUN pip install poetry

#docker secrets
RUN --mount=type=secret,id=ACCESS_TOKEN \
    cat /run/secrets/ACCESS_TOKEN > /usr/src/use_this.env 

    
CMD ls
#CMD ["python3","statistics.py"]   


