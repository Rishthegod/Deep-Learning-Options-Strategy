# AI-Powered Option Trading Strategy 📈

Welcome to the AI-Powered Option Trading Strategy GitHub repository! This project explores the development of an option trading strategy utilizing Python, machine learning, and quantitative finance techniques. Below, is provided an overview of the project structure, key functionalities, and future directions. ➡️


## Table of Contents 📔

1. [Relevance of Options Trading and AI-Driven Strategies](#1-relevance-of-options-trading-and-ai-driven-strategies-) 
2. [Executive Summary and Project Overview](#2-executive-summary-and-project-overview-)
3. [Technologies Used](#3-technologies-used-)
4. [Data Preprocessing](#4-data-preprocessing-)
5. [Feature Engineering](#5-feature-engineering-%EF%B8%8F)
6. [Deep Learning Model Architecture](#6-deep-learning-model-architecture-)
7. [Model Architecture Explained](#7-model-architecture-explained-)
8. [Training and Evaluation](#8-training-and-evaluation-)
9. [Visualizing Model Performance](#9-visualizing-model-performance-)
10. [Options Strategy](#10-options-strategy-)
11. [Backtesting](#11-backtesting-)
12. [Visualizing Strategy Performance](#12-visualizing-strategy-performance-)
13. [Conclusion and Future Directions](#13-conclusion-and-future-directions-%EF%B8%8F)
14. [About Me!](#14-about-me-)

## 1. Relevance of Options Trading and AI-Driven Strategies 📖

Options trading has become a significant part of the modern financial landscape, offering investors and traders opportunities for risk management, income generation, and speculation. According to a report by the Options Clearing Corporation (OCC), the total volume of options contracts traded in 2023 surpassed 11.1 billion, marking a new record and showcasing the growing interest in this market.
As the options trading industry continues to expand, the demand for innovative and data-driven trading strategies has become increasingly important. 

In a rapidly changing market environment, traditional approaches may struggle to keep up with the complexities and volatility of the options market. This is where the application of artificial intelligence (AI) and machine learning techniques can provide a significant advantage.

In a [study](https://www.mdpi.com/1911-8074/16/10/434) published by the Journal of Risk and Financial Management, researchers found that AI-based trading strategies outperformed traditional methods, achieving higher returns and lower risk profiles. The study highlighted the ability of AI algorithms to identify and exploit intricate patterns in vast amounts of financial data, enabling more informed and timely decision-making.

## 2. Executive Summary and Project Overview 🧑‍💻

This project aims to develop an optimal options trading strategy by leveraging deep learning techniques to predict option prices accurately. By combining machine learning algorithms with financial models and risk management principles, the project seeks to generate buy and sell signals that can potentially lead to profitable trading outcomes.

The core of the project revolves around a deep neural network model built using the Keras library. This model is trained on historical data and engineered features, including options Greeks (Delta, Gamma, Theta, Vega) and technical indicators, to accurately predict option prices. The predicted prices are then used to generate trading signals, which form the basis of the options trading strategy.

To ensure the strategy's robustness and efficiency, principles of risk management are incorporated, such as position sizing based on fixed risk per trade and stop-loss orders. Additionally, the strategy accounts for market volatility by adjusting position sizes using the Vega Greek as a volatility metric.

The project follows a structured approach, which includes data preprocessing, feature engineering, model development and training, strategy formulation, and backtesting on historical data. Visualizations are employed throughout the process to gain insights into the model's learning behavior, assess its predictive accuracy, and evaluate the strategy's performance.

The Python Notebook contains all the code for this program, and provides easy-to-understand documentation as well as commentary explaining finance concepts and reasoning to non-technical users. Make sure to install all dependencies before running the notebook.

The presentation will cover the following key areas:
1. Data preprocessing and feature engineering techniques
2. Deep learning model architecture and training process
3. Options strategy formulation, incorporating risk management principles
4. Backtesting and performance evaluation of the strategy
5. Visualizations and interpretations of results

## 3. Technologies Used 🧪

This project leverages various Python libraries and frameworks for data processing, deep learning model development, and visualization. The following technologies were utilized:

1. **Keras**:
   - Keras is a high-level neural networks API, used for building and training the deep learning model in this project.
   - The model architecture was defined using Keras' Sequential API, which allows for easy stacking of layers.
   - Keras was used to define the input layer, dense layers with ReLU activation, dropout layers, and the output layer with linear activation.
   - The model was compiled with the Adam optimizer and mean squared error loss function using Keras.
   - The model training process, including monitoring validation loss, was facilitated by Keras' fit() function.

2. **TensorFlow**:
   - TensorFlow is a popular open-source library for numerical computation and machine learning, serving as the backend for Keras.
   - While Keras provided a high-level interface for model development, TensorFlow handled the low-level computations and optimizations.

3. **NumPy**:
   - NumPy is a fundamental library for scientific computing in Python, used extensively in this project.
   - It was employed for various numerical operations, such as calculating options Greeks (Delta, Gamma, Theta, Vega) using the Black-Scholes model.
   - NumPy's vectorization capabilities were leveraged for efficient computations on large datasets.

4. **Pandas**:
   - Pandas is a powerful data manipulation and analysis library, used for handling and processing the options dataset.
   - It facilitated operations like merging historical asset prices with options data, handling missing values, and applying transformations to the dataset.
   - Pandas also enabled easy indexing and slicing of the dataset during the backtesting process.

5. **Scikit-learn**:
   - Scikit-learn is a machine learning library in Python, providing tools for data preprocessing, model evaluation, and more.
   - In this project, it was used for splitting the dataset into training and testing sets, as well as for scaling the input features using StandardScaler.
   - Scikit-learn's train_test_split function was employed for creating the training and testing datasets.

6. **Matplotlib**:
   - Matplotlib is a plotting library in Python, used for creating static, publication-quality visualizations.
   - It was utilized to generate visualizations of the training and validation loss over epochs, as well as the scatter plot of predicted option prices versus actual prices.

7. **Seaborn**:
   - Seaborn is a data visualization library built on top of Matplotlib, providing a more intuitive and attractive interface for creating statistical graphics.
   - In this project, Seaborn was used for generating the scatter plot of predicted option prices versus actual prices, leveraging its ability to create visually appealing and informative plots.

8. **Yahoo Finance API**:
   - The Yahoo Finance API was used to download historical asset prices and real-time options data for the selected ticker symbols.
   - This data formed the basis for training the deep learning model and backtesting the options trading strategy.

By leveraging these powerful technologies, the project was able to efficiently process and analyze financial data, develop and train a deep learning model for option price prediction, and visualize the results and strategy performance effectively.

## 4. Data Preprocessing 🔮

The initial step in this project involves downloading historical asset prices and real-time options data from a reliable source, the Yahoo Finance API. Acquiring high-quality and up-to-date data is crucial for training the deep learning model effectively and ensuring the backtesting process accurately reflects real-world scenarios.

## 5. Feature Engineering 🛠️

To enhance the predictive capabilities of the deep learning model, a comprehensive set of features was engineered. This includes derived features such as various options Greeks (Delta, Gamma, Theta, Vega) using the widely accepted Black-Scholes model. Additionally, Basic features such as Implied Volatility, DTE, strike-price, and Moneyness are incorporated. The reasoning behind this feature engineering approach is to provide the model with a rich representation of the option's characteristics and the underlying market dynamics, ultimately improving its ability to make accurate predictions.<br>
**In-depth explanations for the Greeks, as well as their formulas, can be found in the Notebook commentary.**<br>
Example Black-Scholes Theta code:
<img width="1209" alt="Screenshot 2024-05-01 at 11 16 05 AM" src="https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/1493c63c-a92a-44db-addd-dd42b7d4e599">


## 6. Deep Learning Model Architecture 💡

The deep learning model architecture employed in this project is a feedforward neural network built using the Keras library. Feedforward neural networks are a class of artificial neural networks where the information flow is unidirectional, from the input layer through the hidden layers to the output layer, without any cycles or loops.

The model architecture consists of the following layers:

1. **Input Layer**: This layer serves as the entry point for the input features. The number of neurons in this layer corresponds to the number of engineered features used for predicting option prices. In our case, the input layer has dimensions equal to the number of features, which include options Greeks (Delta, Gamma, Theta, Vega), moneyness, volatility, and days to expiration.

2. **Dense Layer with ReLU Activation**: The input layer is followed by a dense layer, also known as a fully connected layer. In this layer, each neuron is connected to all the neurons in the previous layer, allowing it to learn complex non-linear relationships within the data. The ReLU (Rectified Linear Unit) activation function is applied to the weighted sum of inputs for each neuron, introducing non-linearity and overcoming the vanishing gradient issue commonly encountered in deep neural networks. The ReLU activation function is defined as:

   ReLU(x) = max(0, x)

   This layer consists of 128 neurons, which can be adjusted based on the complexity of the problem and the available computational resources.

3. **Dropout Layer**: Dropout is a regularization technique used to prevent overfitting, a common issue in deep learning models. During training, dropout randomly deactivates a fraction of neurons (in our case, 20% of neurons) in this layer, effectively creating an ensemble of smaller models. This technique helps the model generalize better to unseen data by reducing the co-adaptation of neurons.

4. **Dense Layer with ReLU Activation**: Another dense layer with 64 neurons and ReLU activation follows the dropout layer. This additional layer further enhances the model's ability to learn intricate patterns and non-linear relationships within the data.

5. **Dropout Layer**: Another dropout layer with a 20% dropout rate is included after the second dense layer to provide additional regularization and improve generalization.

6. **Output Layer with Linear Activation**: The final layer is a dense layer with a single neuron and a linear activation function. This layer outputs the predicted option price, which is a continuous value, making linear activation suitable for regression tasks.

The model architecture can be represented visually as follows:

```
Input Layer (number of features)
    |
Dense Layer (128 neurons, ReLU activation)
    |
Dropout Layer (20% dropout rate)
    |
Dense Layer (64 neurons, ReLU activation)
    |
Dropout Layer (20% dropout rate)
    |
Output Layer (1 neuron, Linear activation)
```

## 7. Model Architecture Explained 🪄

The chosen architecture for the deep learning model is designed to address the specific challenges and requirements of the options trading domain, while adhering to best practices in deep learning.

1. **Dense Layers with ReLU Activation**:
   The dense layers with ReLU activation functions enable the model to learn complex non-linear relationships within the data, which is essential for accurate option price predictions. The ReLU activation function introduces non-linearity by applying the maximum operation between 0 and the weighted sum of inputs for each neuron. This non-linearity is crucial for capturing intricate patterns in the data, as options pricing involves various non-linear factors such as volatility, time decay, and market dynamics.

   The ReLU activation function is defined mathematically as:

   ReLU(x) = max(0, x)

   Where x is the weighted sum of inputs for a given neuron.

   The use of multiple dense layers with ReLU activation allows the model to learn hierarchical representations of the data, with each subsequent layer capturing increasingly abstract and complex features.

2. **Dropout Layers**:
   Overfitting is a common issue in deep learning models, where the model memorizes the training data too well and fails to generalize to unseen data. Dropout layers help mitigate this issue by introducing regularization during the training process.

   The dropout algorithm works as follows:
   - During training, a fraction of neurons (in our case, 20%) are randomly deactivated (set to zero) for each training example.
   - This deactivation is temporary and is applied independently for each training example, creating an ensemble of smaller models.
   - At inference time (when making predictions), all neurons are active, and their outputs are combined to produce the final prediction.

   By randomly deactivating neurons during training, dropout prevents the co-adaptation of neurons, forcing them to learn more robust and generalized representations of the data. This technique effectively regularizes the model, improving its ability to generalize to unseen data and reducing the risk of overfitting.

3. **Linear Output Layer**:
   The linear output layer is particularly well-suited for regression tasks, where the model aims to predict continuous values (option prices) rather than discrete classes. The linear activation function in this layer simply outputs the weighted sum of inputs without applying any non-linearity.

   The linear activation function is defined as:

   Linear(x) = x

   Where x is the weighted sum of inputs for the output neuron.

   By using a linear activation function in the output layer, the model can produce option price predictions that span the entire range of real numbers, without any restrictions or transformations.

The combination of dense layers with ReLU activation, dropout layers for regularization, and a linear output layer creates a powerful deep learning architecture tailored for the task of option price prediction. This architecture allows the model to learn complex non-linear relationships, while also ensuring good generalization and reducing the risk of overfitting.<br>
Model Code snippet:
![Screenshot 2024-05-01 at 1 57 22 PM](https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/4477bb3d-5880-45ac-973e-839b63e70017)

## 8. Training and Evaluation 🏃

To ensure the robustness and reliability of the deep learning model, a rigorous training and evaluation procedure is followed. The available data is split into training and test sets, with the model trained on the training set while monitoring the validation loss to track its performance on unseen data. The evaluation metric employed is the Mean Squared Error (MSE), calculated on the holdout test set. This metric quantifies the average squared difference between the model's predicted option prices and the actual prices, providing a clear measure of its predictive accuracy. The reasoning behind this approach is to maintain a strict separation between training and evaluation data, ensuring an unbiased assessment of the model's generalization capabilities.<br>
Training code snippet:
![Screenshot 2024-05-01 at 2 02 55 PM](https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/addf8c4f-66cc-41e4-b5d1-367d10747b6f)


## 9. Visualizing Model Performance 🔭

To gain insights into the model's learning behavior and assess its predictive accuracy, visualizations play a crucial role. The first visualization depicts the training and validation loss over the epochs during the model training process. This plot helps identify potential issues such as overfitting or underfitting and enables monitoring the model's convergence. The second visualization is a scatter plot comparing the predicted option prices against the actual prices on the test set. This visual representation allows for a direct assessment of the model's predictive performance, with the ideal scenario being a strong positive correlation along the diagonal line.<br>
Validation and Training Loss graph:
![validationloss](https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/02b6ff90-398a-43b0-b819-c50c0181cb54)
Predicted vs. Actual Prices graph:
![Predicted vs actual prices](https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/d2121509-abd3-4928-9fdd-da106563f2eb)


## 10. Options Strategy ♛

Based on the deep learning model's predictions, an options trading strategy is formulated. Historical JPM price data is reindexed to align with the options dataset, and the model is rerun to generate new predictions.<br>
Then, the first step in this strategy is the generation of buy and sell signals by comparing the model's predicted option prices with the current market prices. If the predicted price is higher than the current price, a buy signal is generated, indicating a potential opportunity to purchase the option. Conversely, if the predicted price is lower, a sell signal is generated, suggesting that selling the option might be advantageous.<br>
To manage risk and account for market volatility, the strategy incorporates position sizing and volatility adjustments. Position sizes are determined based on a fixed risk per trade, typically a percentage of the total capital, and a predetermined stop-loss percentage, which limits potential losses. Additionally, the strategy adjusts position sizes based on the option's volatility, as measured by the Vega Greek. Options with higher volatility are allocated smaller position sizes to mitigate risk, while those with lower volatility are allocated larger positions.<br>
Options Strategy code snippet:
![Screenshot 2024-05-01 at 2 11 40 PM](https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/7c05ff73-75ad-414f-8919-9fada746c15c)


## 11. Backtesting 📆

To evaluate the potential performance and profitability of the options trading strategy, backtesting is conducted on historical data. This process involves simulating trades based on the generated buy and sell signals, starting with a specified initial capital. The trades are executed according to the strategy's rules, and the cumulative returns are calculated over time. Backtesting on historical data provides valuable insights into the strategy's behavior under various market conditions and helps identify potential strengths and weaknesses.<br>
Some outputs from backtesting:
![Screenshot 2024-05-01 at 2 16 09 PM](https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/4b39bd13-f761-40d2-ab6f-d9222436c712)


## 12. Visualizing Strategy Performance 💸

To facilitate a clear understanding of the strategy's performance during the backtesting period, a visualization of the cumulative returns is provided. This graph depicts the evolution of the portfolio value over time, reflecting the impact of the executed trades based on the generated signals. The visualization allows stakeholders to quickly grasp the strategy's potential profitability and assess its overall effectiveness.<br>
Simulated Trading Scenario:
![cumulative return](https://github.com/Rishthegod/Deep-Learning-Options-Strategy/assets/42285008/7aebe649-b597-436a-8861-d8075e1c11c4)


## 13. Conclusion and Future Directions 🏎️

In conclusion, this project  demonstrates the application of deep learning techniques to generate an options trading strategy. By leveraging engineered features, a carefully designed deep learning model architecture, and rigorous training and evaluation procedures, the project aims to predict option prices. The predicted prices are then used to generate buy and sell signals, which form the basis of the trading strategy. The strategy's performance is evaluated through backtesting on historical data, with visualizations providing insights into its potential profitability and effectiveness. <br>

This project showcases the potential of combining deep learning and quantitative finance, opening doors for further exploration and refinement. Potential applications of this approach extend beyond options trading, with the possibility of adapting the methodology to other financial instruments or domains. Future improvements could include exploring alternative model architectures, incorporating additional data sources, and enhancing the risk management strategies. Additionally, I am looking into integrating a live paper trading account so the options order can be placed in realtime.

## 14. About Me 👋

Hey there! My name is Rish Sharma, and I'm a passionate Computer Science and Computer Engineering student at the University of California, Irvine. But let me tell you, my interests go way beyond just coding and circuits – I'm fascinated by the intersection of technology and finance, and it's been a driving force for me since my high school days.

I've always been intrigued by the process of finding patterns and extracting insights from vast amounts of data, especially when it comes to the dynamic world of financial markets. To me, the markets are like a complex and ever-evolving puzzle, and the application of artificial intelligence and machine learning techniques offers a powerful approach to unraveling its intricacies.

Throughout my academic journey, I've been on a quest to explore the potential of AI and data-driven strategies in the field of finance. This project is a foray into combining upcoming technologies with financial models to create innovative and potentially 🤞 profitable trading strategies.

If you're as excited as I am about this stuff, or if you have any feedback or collaboration opportunities in mind, feel free to reach out to me at rishits3@uci.edu. I'm always eager to connect with like-minded individuals and explore new possibilities in this ever-evolving field.


---

Thank you for exploring my AI-powered option trading strategy project! I welcome contributions, feedback, and collaboration from the community. Let's continue innovating and exploring the intersection of AI and finance together.
