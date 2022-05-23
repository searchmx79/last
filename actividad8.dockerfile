FROM ubuntu
ENV FLASK_APP=model_api.py
ENV FLASK_ENV = development
ENV FLASK_DEBUG = 0
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt -y upgrade

RUN apt-get install python3-pip -y \
    && pip3 install virtualenv \
    && apt-get install git -y


RUN mkdir analisis \
    && mkdir api

RUN cd analisis
# Creamos el virtualenv llamado analisis
RUN virtualenv analisis
# Activamos el virtualenv
RUN . analisis/bin/activate

RUN apt-get update \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/*

# Descargamos la version mas reciente de Miniconda y ejecutamos el bash
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
# Desactivamos el virtualenv
RUN deactivate

RUN cd ..
RUN cd api

RUN virtualenv api
RUN . api/bin/activate

RUN apt update -y
RUN pip install Flask \
    && pip install flask-swagger-ui

# Descargamos el code que queremos instalar en el docker de un repositorio de git en este caso es el personal
RUN git clone https://github.com/searchmx79/entregable8.git

WORKDIR /entregable8

RUN pip install -r requirements.txt

RUN export FLASK_APP=model_api
RUN export FLASK_ENV=development
RUN export FLASK_DEBUG=0
#Comando para levantar el servidor flask en un container
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

