# **Trainable AI Chatbot Guide**

## **Introduction**
This guide walks you through creating a trainable chatbot using Hugging Face's `transformers` library. It covers setting up an environment, loading a model, training it with custom data, and interacting with the chatbot.

---

## **Step 1: Define the Purpose of Your AI Model**
Before starting, determine:
- The problem your model will solve (e.g., chatbot, image recognition, text classification).
- The type of data required for training.

For this guide, we will focus on creating a **trainable chatbot** using text data.

---

## **Step 2: Choose a Machine Learning Framework**
Popular AI frameworks include:
- **TensorFlow** – Ideal for deep learning applications.
- **PyTorch** – Flexible and widely used for research and experimentation.
- **scikit-learn** – Best for classical machine learning models.
- **Hugging Face Transformers** – Excellent for natural language processing (NLP) tasks.

For a chatbot, we will use **Hugging Face's `transformers` library**, which simplifies working with pre-trained NLP models.

---

## **Step 3: Set Up Your Development Environment**

### **1. Launch an EC2 Instance**
- Log in to AWS and navigate to **EC2**.
- Click **Launch Instance** and choose an **Ubuntu 20.04 LTS AMI**.
- Select an instance type (e.g., `t3.medium` or higher for better performance).
- Turn on "Allow SSH traffic from Anywhere ", "Allow HTTPS traffic from the internet" and "Allow HTTP traffic from the internet".
- Set the storage to 20 GiB of gp3. 
- Configure security groups to allow SSH access.
- Generate and download an SSH key pair for access.

### **2. Connect to Your EC2 Instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### **3. Install Python (if not installed)**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### **4. Create a Virtual Environment (Recommended)**
```bash
python3 -m venv ai_chatbot_env
source ai_chatbot_env/bin/activate
```

### **5. Install Required Libraries**
```bash
pip install transformers datasets torch accelerate
```

---

## **Step 4: Load a Pre-Trained Model**

### **1. Create a Python Script**
Create a new file named `load_model.py`:
```bash
nano load_model.py
```

### **2. Add the Following Code**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a small conversational model (e.g., DialoGPT)
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model and tokenizer loaded successfully.")
```

### **3. Save and Run the Script**
```bash
python load_model.py
```

---

## **Step 5: Fine-Tune the Model with Custom Data**

### **1. Prepare Training Data**
Create a `chat_data.json` file:
```bash
nano /home/ubuntu/chat_data.json
```

Add the following content:
```json
[
    {"input": "Hello!", "response": "Hi! How can I help you?"},
    {"input": "Tell me a joke.", "response": "Why did the chicken cross the road? To get to the other side!"}
]
```

### **2. Create a Training Script**
```bash
nano train_model.py
```

Add the following code:
```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

# Load model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("json", data_files="/home/ubuntu/chat_data.json")

def tokenize_function(examples):
    inputs = [f"User: {i} Bot: {r}" for i, r in zip(examples["input"], examples["response"])]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["input", "response"])

training_args = TrainingArguments(
    output_dir="./trained_chatbot",
    per_device_train_batch_size=2,
    save_strategy="no",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()

model.save_pretrained("./trained_chatbot")
tokenizer.save_pretrained("./trained_chatbot")
```

### **3. Run the Training Script**
```bash
python train_model.py
```

---

## **Step 6: Test the Trained Model**

### **1. Create a Chat Script**
```bash
nano chat_with_bot.py
```

Add the following chatbot interaction script:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load trained model and tokenizer
model_path = "./trained_chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure model uses GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to interact with chatbot
def chat():
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Tokenize input
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

        # Generate response
        output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

        # Decode and print response
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {response}")

# Start chat
if __name__ == "__main__":
    chat()
```

### **2. Run the Chatbot**
```bash
python chat_with_bot.py
```

---

## **Conclusion**
Congratulations! You have successfully created, trained, and deployed a chatbot using Hugging Face’s `transformers` library. You can now fine-tune the model further or deploy it as a service using Flask or FastAPI.



## Improving the dataset
Your chatbot's performance depends on the quality and diversity of the dataset. To make it a great conversationalist, use a large, well-structured dataset.

Option 1: Use Open Datasets
Persona-Chat (Hugging Face Dataset)
ConvAI2 (Hugging Face Dataset)
DailyDialog (Hugging Face Dataset)
OpenAI GPT-3 Conversations (if available)
To load an open dataset:

python
Copy
Edit
from datasets import load_dataset

dataset = load_dataset("daily_dialog")  # Replace with chosen dataset
print(dataset)
Option 2: Create a Custom Dataset
If you want a highly personalized chatbot, create a JSON dataset with structured conversational exchanges:

json
Copy
Edit
[
    {"input": "Hey, how are you?", "response": "I'm great! How about you?"},
    {"input": "What's your favorite book?", "response": "I love 'The Hitchhiker's Guide to the Galaxy'!"},
    {"input": "Tell me a fun fact!", "response": "Did you know that honey never spoils?"}
]
Save this as custom_chat_data.json.

## Choose a more advanced model
 - DialoGPT (Small/Medium/Large) – Optimized for conversational AI.
 - GPT-2 / GPT-3 – Powerful, requires more compute resources.
 - Mistral-7B / LLaMA-2 – Good balance between size and capability.
To load a larger model:
--python--
model_name = "microsoft/DialoGPT-medium"  # Change to a better model in load_model.py









Option 2: Create a Web Interface
To make the chatbot accessible via a browser or API, consider using Flask or FastAPI.
Using Flask
Install Flask:

bash
Copy
Edit
pip install flask
Create a server.py file:

bash
Copy
Edit
nano server.py
Add the following code:

python
Copy
Edit
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_path = "./trained_chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
Run the Flask server:

bash
Copy
Edit
python server.py
Test your chatbot using Postman or cURL:

bash
Copy
Edit
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"message": "Hello!"}'
Option 3: Deploy as a Telegram or Discord Bot
You can integrate the chatbot into platforms like Telegram or Discord by using their APIs.
Option 4: Deploy to AWS Lambda or Google Cloud Functions
To make it serverless, deploy the chatbot model to AWS Lambda using Amazon API Gateway or Google Cloud Functions.


Check Available Disk Space
Run the following command to check disk usage:

bash
Copy
Edit
df -h
If your disk is nearly full, you’ll need to free up space.

## Debugging
# Remove Unnecessary Files if storage is filled
You can remove older checkpoints and cached models that take up space:
Delete Cached Hugging Face Models
bash
Copy
Edit
rm -rf ~/.cache/huggingface
Delete Unused PyTorch Checkpoints
bash
Copy
Edit
rm -rf ./chatbot_model/checkpoint-*
Clear APT Cache
bash
Copy
Edit
sudo apt clean
Step 3: Expand Your Storage (Optional)
If using AWS, consider increasing the instance’s disk size via EBS Volume Expansion in the AWS EC2 settings.






Startup guide: 
python load_model.py
python train_model.py
python chat_with_bot.py
