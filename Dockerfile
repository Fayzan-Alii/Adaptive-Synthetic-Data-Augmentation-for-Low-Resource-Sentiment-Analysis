# Use a recent PyTorch image with CUDA 12 (for torch==2.4.1)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/

# Install build tools for fasttext and other native dependencies
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN mkdir -p /app/Models /app/Data /app/Images

ENTRYPOINT ["python", "generate.py"]