FROM jupyter/tensorflow-notebook

LABEL maintainer="lainisourgod"

USER root

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get autoremove -y

RUN pip install joblib==0.13.2 \
                scikit-learn==0.19.2 \
                flask

USER 1000

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/)"

EXPOSE 5000

ENTRYPOINT ["python3", "server/app.py"]
