FROM syzygianinfern0/stormbase:latest

# Set working directory and copy your app
WORKDIR /app
COPY . /app

# Expose Gradio port
EXPOSE 7860

# Run your Gradio app
CMD ["python3", "evaluate_demo.py"]
