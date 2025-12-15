
## Predictive Maintenance Time series forecasting by Traditional Method using Random Forest

This provides real-time machine failure predictions along with a 10-day failure forecast summary. Users can upload their own dataset in Excel format (up to 200MB per file) to automatically process the data, train the failure prediction model, and visualize key metrics.

<img width="1817" height="801" alt="Dashboard" src="https://github.com/user-attachments/assets/415f99ee-ccff-40cc-ae01-684bb6b9e1bd" />

---

## Dataset Uploading

<img width="1824" height="834" alt="Dataset_and_Metrics" src="https://github.com/user-attachments/assets/cb1a926c-f55e-42b6-9a61-bc204ad86aae" />


After uploading a dataset such as dataset_with_timestamp.xlsx, the system loads the file, displays a preview of the uploaded dataset, and shows the processed dataset after cleaning, encoding, and scaling.

The dashboard then evaluates the trained model and reports key metrics such as ROC AUC, PR AUC, and a detailed classification report. For the example dataset, the model achieved a ROC AUC of 0.9818 and a PR AUC of 0.9605. The classification results show very high prediction accuracy, with f1-scores of 0.9982 for normal operation (class 0) and 0.9489 for failure (class 1), resulting in an overall accuracy of 99.65% on the test set.

---
## 10 days Prediction table

<img width="1782" height="571" alt="10days_prediction" src="https://github.com/user-attachments/assets/63c1bdff-2adc-4993-85b1-cf010cd56ee6" />

The dashboard also generates a 10-day failure forecast summary. Based on the predictions, the system estimates how many machines are likely to fail on each future date. In the example run, the model predicted 349 machines to fail within the next 10 days. This forecasting capability helps maintenance teams plan ahead and prioritize inspections or repairs.

---
## Graphical Representation

<img width="911" height="682" alt="Graphical_representation" src="https://github.com/user-attachments/assets/9095cc5b-8936-4feb-84ef-147d5ff5a368" />

The tool is designed to serve as an interactive predictive maintenance solution, enabling users to quickly analyze machine health, identify risks, and take proactive actions to reduce downtime.
