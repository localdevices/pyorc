FROM continuumio/miniconda3
COPY envs/pyorc-dev.yml .

COPY . .
RUN apt-get update
RUN apt install libgl1-mesa-glx ffmpeg libsm6 libxext6 -y
RUN conda install -c conda-forge mamba
RUN mamba install -c conda-forge xarray geopandas cartopy
RUN mamba update -c conda-forge -f pyorc-dev.yml
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "pyorc-dev", "pip", "install", "-e", "."]
RUN pip install -e .
EXPOSE 5003
# The code to run when container is started:
CMD ["pyorc"]


#RUN pip install -e .
#CMD ["bash"]