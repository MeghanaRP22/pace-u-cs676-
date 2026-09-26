FROM python:3.11-slim

# Spaces runs your container as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# Copy requirements first so Docker can cache the install layer.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# main.py, credibility.py, model_weights.json and data/domain_pc1.csv are all
# needed at run time; .dockerignore keeps secrets and the venv out.
COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "main.py", \
     "--server.address=0.0.0.0", "--server.port=7860"]
