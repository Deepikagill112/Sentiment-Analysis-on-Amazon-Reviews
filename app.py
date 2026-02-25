from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    
    if text:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return render_template('result.html', text=text, sentiment=sentiment, polarity=polarity)
    
    return render_template('landing.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # TSV और CSV दोनों को handle करो
        if file.filename.endswith('.tsv'):
            df = pd.read_csv(file, sep='\t')
        else:
            df = pd.read_csv(file)
        
        sentiments = []
        
        for text in df.iloc[:, 0]:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0:
                sentiments.append('Positive')
            elif polarity < 0:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        
        # Graph बनाओ
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        plt.figure(figsize=(8, 6))
        sentiment_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#3498db'])
        plt.title('Sentiment Analysis Results')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Image को base64 में convert करो
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('graph_result.html', graph=graph_url, sentiments=sentiments)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)