# Build Your Own AI Model Using Python

This project allows you to build a simple AI model using Python and train it on data you collect from the internet. Follow the steps outlined below to create and test your own AI model.

---

## Project Overview

### File 1: **Data Collection**

- Use this file to collect data directly from Google.
- How it works:
  1. Run the file and select **Option 1**.
  2. Enter the topic or keywords you want data for (e.g., articles, tutorials, etc.).
  3. The script will fetch relevant data from Google, filter it, and save it in a file called `data.txt`.

---

### File 2: **Model Training**

- This file trains your AI model using the data collected in `data.txt`.
- **Requirements**:
  - Install the `torch` library before running this file.
- How it works:
  1. Run the file to start training your model.
  2. The training process uses `data.txt` to build the AI model.
  3. Once training is complete, the model will be saved as `simple_lstm_model.pth`.

---

### File 3: **Model Testing**

- Use this file to test and interact with your trained AI model.
- How it works:
  1. Run the file and load the `simple_lstm_model.pth` model.
  2. Start a conversation with your model to see how well it performs.
  3. Type your messages, and the model will generate responses.

---

## Important Notes

1. **Data Volume**:

   - Make sure the training dataset (`data.txt`) is sufficiently large and diverse.
   - A small dataset will result in a poorly trained model that won't perform well.

2. **Use Case**:
   - This project is a starting point for creating your own AI model. Customize it further to suit your specific needs.

---

## Next Steps

1. Collect a large dataset for better training results.
2. Experiment with different architectures in the training file to improve performance.
3. Share and showcase your trained AI model as your personal project!

Start building your AI model today and make it uniquely yours!
