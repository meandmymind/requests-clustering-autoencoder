FROM jupyter/tensorflow-notebook

MAINTAINER lainisourgod

USER root

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get autoremove -y \
&& pip install joblib scikit-learn

USER 1000

COPY . .

ENTRYPOINT ["jupyter", "notebook"]
