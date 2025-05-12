# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app/

# Install necessary packages
RUN pip install --no-cache-dir \
    transformers==4.31.0 \
    datasets==2.14.0 \
    trl==0.7.1 \
    fasttext==0.9.2 \
    nltk==3.8.1 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    rouge_score==0.1.2

# Download NLTK resources
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Set up environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Make a directory for model outputs and data
RUN mkdir -p /app/Models /app/Data /app/Images

# Command to run the inference script
ENTRYPOINT ["python", "inference.py"]