The task was performed using a linear regression model from the sklearn library
and using a neural network model with fully connected Dense layers from the Keras library.

The linear model showed the following metrics:
- Model accuracy: 0.4384316213369278
- R2score 0.4384316213369278
- MSE 2992.5812293010176
- RMSE 54.704490028708044

Keras model (based on Dense layers) showed following metrics:
- Mean MSE: 3095.037841796875
- Mean MAE: 42.33686828613281
  
![Metrics](https://github.com/user-attachments/assets/208bec0e-eb12-4193-bfe5-71496998361e)

![Predicted_True_Difference](https://github.com/user-attachments/assets/9aa3deac-b7bb-4070-8114-1a5a6e8a61c3)


Keras model (based on Dense layers with Cross-Validation) showed following metrics from four splits:
- Mean MSE: 3021.351806640625
- Mean MAE: 44.45400524139404
  
![KFold_metrics](https://github.com/user-attachments/assets/a9ce0dd9-6976-4c95-9041-d37f8729d65e)

![KFold_predicted_true_difference](https://github.com/user-attachments/assets/2d5cd956-d1f7-4d7f-8a15-53b7b385b6ee)


