FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
LABEL authors="flateon"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]