FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pyhydra

# Install Python dependencies (cached layer)
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade setuptools && \
    pip install --no-cache-dir --no-build-isolation pySWATPlus && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Install CoSMoS spatial field generation (distributed as a wheel)
COPY docker/cosmos_py-0.1.0-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/cosmos_py-0.1.0-py3-none-any.whl

# Copy the package and install it
COPY . .
RUN pip install --no-cache-dir -e .

# Jupyter: disable token/password for local use, allow all origins
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/pyhydra/notebooks"]
