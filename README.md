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

&#x20;                   E-COMMERCE ORDER

&#x20;                          │

&#x20;            ┌─────────────┴─────────────┐

&#x20;            │                           │

&#x20;            ▼                           ▼

&#x20;      Product Image                Order Data

&#x20;            │                           │

&#x20;            ▼                           ▼

&#x20;       ResNet-18                 Return Risk Model

&#x20;            │                           │

&#x20;            ▼                           ▼

&#x20;    Product Classification       Return Probability

&#x20;            │                           │

&#x20;            └─────────────┬─────────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;                ORDER INTELLIGENCE

&#x20;                          │

&#x20;            ┌─────────────┴─────────────┐

&#x20;            │                           │

&#x20;            ▼                           ▼

&#x20;      Product Class              Return Risk

&#x20;      Confidence                 HIGH / LOW

