import os
import sys
import re
import joblib
# ⭐️ 중요: 상단에 render_template이 반드시 포함되어 있어야 HTML을 읽어옵니다!
from flask import Flask, request, jsonify, render_template
from konlpy.tag import Okt

app = Flask(__name__)

# 📍 [절대경로 세팅] 파일 위치 자동 추적
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'movie_sentiment_model.pkl')

# 📍 [Okt 토크나이저 함수 정의] joblib.load 보다 위에 위치해야 함
def okt_tokenizer(text):
    return okt.morphs(text)

# 📍 [자바 및 Okt 시동] 예외 방어 처리
try:
    print("⚙️ 1. 시스템 기본 자바 엔진 시동 시도 중...")
    okt = Okt()
except Exception as e:
    print("⚠️ 시스템 자바 자동 인식을 실패하여 내 PC용 우회 경로를 가동합니다.")
    # 본인 PC 환경에 맞는 jvm.dll 주소인지 꼭 확인하세요!
    ms_jvm_path = r"C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot\bin\server\jvm.dll"
    
    if not os.path.exists(ms_jvm_path):
        print(f"❌ [에러] 자바 구동 파일({ms_jvm_path})을 찾을 수 없어 서버를 가동할 수 없습니다.")
        sys.exit(1)
        
    okt = Okt(jvmpath=ms_jvm_path)

# 📍 [인공지능 장비 로드]
print("⚙️ 2. 인공지능 모델 및 번역기 절대경로 로드 중...")
tfidf = joblib.load(TFIDF_PATH)
model = joblib.load(MODEL_PATH)
print("🚀 모든 장비 장착 완료! Flask 서버를 가동합니다.")


# ----------------------------------------------------
# ⭐️ [복구 완료] 메인 홈페이지 화면 로드 (GET 방식)
# ----------------------------------------------------
@app.route('/')
def home():
    # templates 폴더 안의 index.html을 브라우저에 그려줍니다.
    return render_template('index.html')


# ----------------------------------------------------
# 📍 실전 실시간 추론 API 엔드포인트 (POST 방식)
# ----------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_review = data.get('review', '')
    
    if not user_review:
        return jsonify({'error': '리뷰 내용이 비어있습니다.'}), 400
        
    # 데이터 전처리 (한글만 추출)
    cleaned = [" ".join(re.compile(r'[ㄱ-ㅣ가-힣]+').findall(user_review))]
    
    # 추론 연산
    X_tfidf = tfidf.transform(cleaned)
    prediction = int(model.predict(X_tfidf)[0]) # 0(부정) 또는 1(긍정)
    prob = model.predict_proba(X_tfidf)[0][prediction] # 확률 값
    
    # 결과 반환
    return jsonify({
        'sentiment': '긍정' if prediction == 1 else '부정',
        'confidence': round(float(prob) * 100, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)