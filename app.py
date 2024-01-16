import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from transformers import pipeline
import time
import base64

headers = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# 指定されたURLからレビューを取得し、感情分析を行う関数
def get_reviews(url, ASIN):
    # 指定されたURLからHTMLを取得
    r = requests.get(url, headers=headers)
    # 取得したHTMLをパース
    soup = BeautifulSoup(r.text, 'html.parser')
    # レビュー部分を取得
    reviews = soup.find_all(attrs={'data-hook': 'review'})
    review_list = []
    # 感情分析のパイプラインを作成
    sentiment_analysis = pipeline('sentiment-analysis', model="cl-tohoku/bert-base-japanese-whole-word-masking")
    for review in reviews:
        # レビューの本文を取得
        review_body = review.find(attrs={'data-hook': 'review-body'}).text.strip()
        review_body = ' '.join(review_body.split())
        review_body = ''.join(c if c.isalnum() or c.isspace() else '。' for c in review_body)
        # レビューの長さを512に制限
        # review_body = review_body[:512]
        # 星評価を取得
        star_rating = review.find('span', {'class': 'a-icon-alt'}).text
        star_rating = star_rating[-3:]
        star_rating = float(star_rating)
        # 感情分析を実行
        sentiment = sentiment_analysis(review_body)[0]
        # レビューの本文、星評価、感情スコアをリストに追加
        review_list.append([review_body, star_rating, round(sentiment['score'], 4)])
    return review_list

st.title('Amazon Review Sentiment Analysis App')

# ASINコードの入力を受け付けるテキストボックスを作成
ASIN = st.text_input('ASINコードを入力してください')
st.text('最大20件まで表示します。')

# 列名を初期化
df = pd.DataFrame(columns=['review_body', 'star_rating', 'sentiment_score'])  
if st.button('感情分析を実行'):
    for i in range(1, 3):
        # レビューページのURLを作成
        url = f'https://www.amazon.co.jp/product-reviews/{ASIN}/?pageNumber={i}'
        # レビューを取得
        reviews = get_reviews(url, ASIN)
        if reviews:  # レビューが存在する場合のみ追加
            # 取得したレビューをデータフレームに変換
            df_temp = pd.DataFrame(reviews, columns=['review_body', 'star_rating', 'sentiment_score'])
            # データフレームを結合
            df = pd.concat([df, df_temp], ignore_index=True)
        time.sleep(3)

    # sentiment_scoreを降順にソート
    df = df.sort_values(by='sentiment_score', ascending=False)  
    st.write(df)
    if not df.empty:
        # データフレームをCSVに変換
        csv = df.to_csv(index=False)
        # 文字列をバイトに変換し、Base64にエンコード
        b64 = base64.b64encode(csv.encode()).decode()  
        # CSVダウンロードリンクを作成
        href = f'<a href="data:text/csv;base64,{b64}" download="reviews.csv">CSVをダウンロード</a>'
        st.markdown(href, unsafe_allow_html=True)

