# Flipkart Order Intelligence & AI Support Agent

An end-to-end AI/ML project for e-commerce order intelligence that combines machine learning, computer vision, retrieval-augmented generation (RAG), and an agentic workflow built with LangGraph.

The project contains three major AI capabilities:

1. **Return-Risk Prediction** - predicts the probability that an order will be returned using customer and order features.
2. **Product Image Classification** - classifies product images using a fine-tuned ResNet-18 model.
3. **AI Support Agent** - uses LangGraph to route customer queries across return-risk prediction, image classification, policy retrieval, safety guardrails, and conversation context.

---

## Project Overview

Modern e-commerce support systems need to work with multiple types of information, including structured order data, product images, company policies, and conversational customer requests.

This project demonstrates how these capabilities can be combined into one AI system.

### Part 1 - Return-Risk Prediction

A machine learning pipeline predicts the probability that an order will be returned.

The pipeline includes:

- Synthetic e-commerce order data
- Data preprocessing
- Logistic Regression baseline
- Random Forest model
- Hyperparameter tuning
- ROC-AUC evaluation
- Threshold selection
- Feature importance analysis
- Saved production inference model

### Part 2 - Product Image Classification

A computer vision pipeline classifies product images into Fashion-MNIST categories.

The pipeline includes:

- Transfer learning with ResNet-18
- Grayscale-to-RGB preprocessing
- ImageNet normalization
- Model training and evaluation
- Per-class prediction confidence
- Saved PyTorch model
- Sample images for inference

### Part 3 - Agentic AI Support Assistant

A LangGraph-based support agent integrates the ML models with a RAG knowledge base.

The agent can:

- Answer return-policy questions using semantic retrieval
- Predict order return risk using the Part 1 model
- Classify product images using the Part 2 model
- Detect basic prompt-injection attempts
- Reject policy answers when retrieval confidence is insufficient
- Preserve order context for follow-up queries
- Route requests through a stateful LangGraph workflow
- Produce deterministic responses using a MOCK_LLM layer

---

## System Architecture

```text
                         CUSTOMER QUERY
                               |
                               v
                    +----------------------+
                    |   LangGraph Router   |
                    +----------------------+
                               |
          +--------------------+--------------------+
          |                    |                    |
          v                    v                    v
   Policy Question       Return-Risk Query      Image Query
          |                    |                    |
          v                    v                    v
   RAG Knowledge Base    Part 1 ML Model      Part 2 ResNet-18
          |                    |                    |
          v                    v                    v
   FAISS Retrieval       Return Probability    Product Category
          |                    |                    |
          +--------------------+--------------------+
                               |
                               v
                     +-------------------+
                     | Response / MOCK   |
                     |       LLM         |
                     +-------------------+
                               |
                               v
                         FINAL ANSWER

Additional controls:
- Prompt-injection guardrail
- RAG confidence threshold
- Conversation/order context
```

---

## Tech Stack

### Machine Learning
- Python
- Scikit-learn
- Random Forest
- Logistic Regression
- Joblib

### Deep Learning & Computer Vision
- PyTorch
- Torchvision
- ResNet-18
- Fashion-MNIST
- Pillow

### Agentic AI
- LangGraph
- Stateful routing
- Tool-based model invocation
- MOCK_LLM deterministic response layer

### RAG
- Sentence Transformers
- FAISS
- Semantic embeddings
- Similarity-based retrieval
- Groundedness thresholding

### Safety
- Prompt-injection detection
- Retrieval-confidence guardrail
- Controlled agent routing

### Testing & Development
- Pytest
- Git
- Git LFS
- Conda / Python virtual environment

---

## Project Structure

```text
flipkart-order-intelligence/
|
|-- data/
|   `-- sample_images/
|       |-- 00_ankle_boot.png
|       |-- 01_pullover.png
|       |-- 02_trouser.png
|       |-- 03_trouser.png
|       `-- 04_shirt.png
|
|-- models/
|   |-- product_classifier.pt
|   |-- return_risk_model.pkl
|   `-- return_risk_threshold.txt
|
|-- part1_return_risk/
|   |-- generate_orders.py
|   |-- inspect_data.py
|   |-- orders_dataset.csv
|   `-- train_return_risk.py
|
|-- part2_image_classifier/
|   `-- train_product_classifier.py
|
|-- part3_prediction/
|   `-- predict.py
|
|-- part3_support_agent/
|   |-- agent.py
|   |-- mock_llm.py
|   |-- rag.py
|   |-- safety.py
|   |-- tools.py
|   `-- knowledge_base/
|       |-- 01_apparel_footwear_returns.txt
|       |-- 02_electronics_returns.txt
|       |-- ...
|       `-- 12_refund_timeline.txt
|
|-- tests/
|   `-- test_support_agent.py
|
|-- .gitattributes
|-- .gitignore
|-- README.md
`-- requirements.txt
```

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/s571662/flipkart-order-intelligence.git
cd flipkart-order-intelligence
```

### 2. Create a Python Environment

Using Conda:

```bash
conda create -n flipkart-ai python=3.11 -y
conda activate flipkart-ai
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the Agent

```bash
python -c "from part3_support_agent.agent import agent_graph; print('Agent graph compiled successfully')"
```

Expected output:

```text
Agent graph compiled successfully
```

### 5. Run Automated Tests

```bash
python -m pytest tests/test_support_agent.py -q
```

Expected result:

```text
4 passed
```

---

## Usage Examples

### Policy RAG Query

```bash
python -c "from part3_support_agent.agent import agent_graph; result=agent_graph.invoke({'user_query':'What is the return policy for electronics?'}); print(result['response']['final_answer'])"
```

Example output:

```text
Electronics products may be returned within 7 days of delivery if the item is defective, damaged, or not as described.
```

### Return-Risk Prediction

```bash
python -c "from part3_support_agent.agent import agent_graph; order={'price_inr':2499,'delivery_distance_km':12.5,'customer_tenure_days':450,'delivery_days':5,'discount_pct':20,'num_previous_orders':8,'num_previous_returns':2,'rating_given':4,'payment_method':'COD','product_category':'Footwear','is_weekend_order':1}; result=agent_graph.invoke({'user_query':'What is the return risk for this order?','order_features':order}); print(result['response']['final_answer'])"
```

Example output:

```text
Return probability is 57.05%. Risk bucket: Medium.
```

### Product Image Classification

```bash
python -c "from part3_support_agent.agent import agent_graph; result=agent_graph.invoke({'user_query':'Classify this product image','image_path':'data/sample_images/00_ankle_boot.png'}); print(result['response']['final_answer'])"
```

Example output:

```text
Predicted product category: Ankle boot. Confidence: 96.75%.
```

### Conversation Context

The agent can preserve an order ID in its state and use it in a follow-up request:

```bash
python -c "from part3_support_agent.agent import agent_graph; result=agent_graph.invoke({'user_query':'What about its delivery?','current_order_id':'1523'}); print(result['response']['final_answer'])"
```

Example output:

```text
You are asking about order 1523. The order ID was preserved from the current conversation state.
```

---

## Safety and Groundedness

The support agent includes lightweight safety and reliability controls.

### Prompt-Injection Guardrail

Incoming queries are checked for suspicious instructions that attempt to override the assistant's intended behavior. Blocked requests are stopped before reaching downstream tools.

### RAG Groundedness

Policy answers use semantic retrieval from the local knowledge base. A similarity threshold of `0.60` is used to prevent the agent from confidently answering questions when supporting policy evidence is insufficient.

For example, an unsupported product-policy query should return:

```text
grounded: False
```

rather than generating an unsupported policy answer.

### Deterministic MOCK_LLM

The project uses a deterministic `MOCK_LLM` response layer by default. This keeps the project:

- Reproducible
- Testable
- Independent of paid API keys
- Easy to evaluate locally

The architecture can later be extended to use a hosted LLM while retaining the same LangGraph workflow and tool interfaces.

---

## Automated Testing

The end-to-end test suite currently verifies:

- Policy RAG routing and grounded responses
- Return-risk model invocation
- Product-image classification
- Conversation-state handling

Run:

```bash
python -m pytest tests/test_support_agent.py -q
```

Current result:

```text
4 passed
```

---