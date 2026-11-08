"""Generate small, self-authored sample PDFs (our own text, so freely redistributable)."""
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

DOCS = {
"ml_basics.pdf": ("Machine Learning Basics", [
 ("What is Machine Learning?", [
  "Machine learning is a field of artificial intelligence in which computers learn patterns from data instead of following hand-written rules. A model is trained on examples and then used to make predictions on new data.",
  "Supervised learning is the most common type of machine learning. In supervised learning, the training data contains inputs together with the correct answers, called labels. The model learns the mapping from inputs to labels. Example: predicting whether an email is spam from emails that people already marked as spam or not spam.",
  "Unsupervised learning works with data that has no labels. The model looks for structure by itself, such as grouping similar customers into clusters."]),
 ("Classification and Regression", [
  "Classification and regression are the two main kinds of supervised learning problems. Classification predicts a category, for example spam or not spam, or the digit shown in an image. Regression predicts a continuous number, for example the price of a house or tomorrow's temperature.",
  "The key difference between classification and regression is the type of output: classification outputs a discrete class label, while regression outputs a real number. Classification is usually measured with accuracy, precision and recall. Regression is usually measured with mean squared error."]),
 ("Random Forest", [
  "A random forest is an ensemble of many decision trees. Each tree is trained on a random sample of the training rows and considers only a random subset of the features at every split. This randomness makes the trees different from each other.",
  "To make a prediction, every tree votes and the forest returns the majority class for classification, or the average value for regression. Averaging many different trees reduces overfitting compared to a single deep decision tree. Random forests work well on tabular data with little tuning."]),
 ("Gradient Descent", [
  "Gradient descent is an optimisation algorithm used to train models. Imagine standing on a foggy hill and wanting to reach the valley. You feel the slope under your feet and take a step downhill. Repeating this brings you towards the bottom.",
  "In machine learning the hill is the loss function, which measures how wrong the model is. The gradient tells which direction increases the loss, so we move the parameters in the opposite direction. The learning rate controls the size of each step. If the learning rate is too large the model can overshoot the minimum; if it is too small, training is very slow."]),
]),
"ai_overview.pdf": ("Artificial Intelligence Overview", [
 ("Artificial Intelligence", [
  "Artificial intelligence, or AI, is the broad goal of building computer systems that perform tasks that normally need human intelligence, such as understanding language, recognising images and making decisions. Machine learning is a subset of AI, and deep learning is a subset of machine learning."]),
 ("Neural Networks and Deep Learning", [
  "Deep learning uses neural networks with many layers. A neural network is made of neurons that take weighted inputs, add a bias and apply an activation function. Training adjusts the weights using backpropagation, which computes gradients of the loss with respect to every weight.",
  "Deep learning is especially strong on images, audio and text, where hand-designing features is difficult."]),
 ("Transformers, LLMs and RAG", [
  "The Transformer is a neural network architecture built around self-attention. Self-attention lets every word in a sentence look at every other word and decide which ones are important for its meaning. Transformers were introduced in 2017 and are the foundation of modern large language models.",
  "A large language model, or LLM, is a Transformer trained on huge amounts of text to predict the next token. LLMs can hallucinate, which means they sometimes state false information confidently.",
  "Retrieval-Augmented Generation, or RAG, reduces hallucination by first retrieving relevant passages from a trusted document collection and then asking the LLM to answer using only those passages. RAG also lets a system use private or recent documents without retraining the model."]),
]),
"python_primer.pdf": ("Python Primer for Data Work", [
 ("Variables, Types and Functions", [
  "Python is a popular programming language for data science. A variable is a name that refers to a value, for example age = 25. Common built-in types are int, float, str, bool, list, tuple, set and dict.",
  "A function is defined with the def keyword. Functions group reusable logic and can return values. Example: def square(x): return x * x."]),
 ("Lists, Dictionaries and Comprehensions", [
  "A list is an ordered, changeable collection written with square brackets. A dictionary stores key-value pairs written with curly braces. A list comprehension builds a new list in one line, for example squares = [x * x for x in range(5)].",
  "Exceptions are handled with try and except blocks. Files are best opened with the with statement, which closes the file automatically even if an error occurs."]),
]),
"data_science_workflow.pdf": ("The Data Science Workflow", [
 ("Workflow Overview", [
  "A typical data science project follows these steps: define the problem, collect data, clean the data, explore the data, engineer features, train models, evaluate them and deploy the best one. The steps are repeated many times rather than done once."]),
 ("Data Cleaning and Missing Values", [
  "Real data is messy. Common problems are duplicate rows, wrong data types, outliers and missing values. Missing values can be handled by dropping rows, filling with the mean or median, or using a model-based imputer. The right choice depends on how much data is missing and why.",
  "Feature engineering is the process of creating new input columns that help a model learn, for example extracting the month from a date or computing the ratio of two columns. Good features often matter more than the choice of algorithm."]),
 ("Model Evaluation", [
  "Data is split into training, validation and test sets. The model learns from the training set, the validation set is used to tune settings, and the test set is used once at the end to estimate real-world performance. Overfitting happens when a model scores well on training data but poorly on new data. Cross-validation repeats the split several times to give a more reliable estimate."]),
]),
"notely_api_guide.pdf": ("Notely API User Guide (fictional example)", [
 ("Authentication and Limits", [
  "Notely is a fictional note-taking service used only to demonstrate technical documentation search. All requests to the Notely API must include an API key in the X-Notely-Key header. Keys can be created in the dashboard under Settings, then Developer.",
  "The default rate limit is 60 requests per minute per key. When the limit is exceeded the API returns HTTP status 429 and the Retry-After header tells the client how many seconds to wait. Notes are created with POST /v1/notes and the maximum note size is 100 KB."]),
]),
}


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    for fname, (title, pages) in DOCS.items():
        story = []
        for i, (heading, paras) in enumerate(pages):
            if i:
                story.append(PageBreak())
            story.append(Paragraph(heading, ss["Heading1"]))
            story.append(Spacer(1, 12))
            for p in paras:
                story.append(Paragraph(p, ss["BodyText"]))
                story.append(Spacer(1, 8))
        doc = SimpleDocTemplate(str(out_dir / fname), pagesize=A4, title=title, author="RAG-Document-Intelligence project (self-generated)")
        doc.build(story)
        print("created", fname, f"({len(pages)} pages)")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "sample_pdfs")
