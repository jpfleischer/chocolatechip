FROM python:3.12

WORKDIR /usr/src

RUN apt update 

COPY statistics.py ./

#cloning and such (irrelevant with volume mount)
#   RUN apt install git -y
#   RUN git clone https://github.com/jpfleischer/chocolatechip.git

#need to move to makefile
#   RUN cd chocolatechip; pip install -r requirements.txt
#   RUN cd chocolatechip; pip install -e . --config-settings editable_mode=strict

RUN pip install --upgrade pip

#fixing some errors
RUN apt install nano
RUN pip install poetry

#docker secrets
RUN --mount=type=secret,id=ACCESS_TOKEN \
    cat /run/secrets/ACCESS_TOKEN > /usr/src/use_this.env 
    
CMD ls
#CMD ["python3","statistics.py"]   

# Once you make:
#   pip install -r requirements.txt
#   pip install -e . --config-settings editable_mode=strict in chocolatechip

