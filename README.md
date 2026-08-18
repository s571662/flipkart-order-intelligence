\# 🛍️ Flipkart Order Intelligence



An end-to-end AI/ML pipeline for e-commerce order intelligence that combines \*\*product image classification\*\* with \*\*customer return-risk prediction\*\*.



The system analyzes an order from two perspectives:



1\. 🖼️ \*\*Product Image Classification\*\* — identifies the product category using a ResNet-18 deep learning model.

2\. 📊 \*\*Return Risk Prediction\*\* — estimates the probability that an order will be returned using a machine learning classification model.



The final pipeline combines both predictions into a single order intelligence result.



\---



\## 🚀 Project Overview



E-commerce platforms handle millions of orders across different product categories and customer segments.



This project demonstrates how machine learning can be used to:



\- Automatically identify products from images

\- Predict the probability of an order being returned

\- Classify orders into high/low return-risk categories

\- Combine computer vision and tabular ML predictions into a unified intelligence pipeline



\---



\## 🏗️ Architecture



```text

                  E-COMMERCE ORDER
                          │

            ┌─────────────┴─────────────┐

            │                           │

            ▼                           ▼

      Product Image                Order Data

            │                           │

            ▼                           ▼

       ResNet-18                 Return Risk Model

            │                           │

            ▼                           ▼

    Product Classification       Return Probability

            │                           │

            └─────────────┬─────────────┘

                          │

                          ▼

                ORDER INTELLIGENCE

                         │

            ┌─────────────┴─────────────┐
            │                           │

            ▼                           ▼

      Product Class              Return Risk

      Confidence                 HIGH / LOW

## ✨ Features

- Product classification using ResNet-18
- Fashion-MNIST based product categories
- Return-risk prediction using tabular customer/order data
- Probability-based return-risk scoring
- Configurable return-risk threshold
- Combined image + tabular ML prediction pipeline
- Command-line prediction interface
- PyTorch model saved as a Git LFS artifact
- Reproducible Python environment using `requirements.txt`

---

## 🛠️ Tech Stack

### Programming
- Python

### Machine Learning
- Scikit-learn
- PyTorch
- Torchvision

### Data Processing
- Pandas
- NumPy

### Computer Vision
- PIL / Pillow
- ResNet-18
- Fashion-MNIST

### Model Persistence
- Joblib
- PyTorch `.pt` model
- Git LFS

---

## 📁 Project Structure

```text
flipkart-order-intelligence/
│
├── data/
│   └── sample_images/
│       └── test_image.png
│
├── models/
│   ├── product_classifier.pt
│   ├── return_risk_model.pkl
│   └── return_risk_threshold.txt
│
├── part1_return_risk/
│   ├── generate_orders.py
│   ├── inspect_data.py
│   ├── orders_dataset.csv
│   └── train_return_risk.py
│
├── part2_image_classifier/
│   └── train_product_classifier.py
│
├── part3_prediction/
│   └── predict.py
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt