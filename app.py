# app.py
from flask import Flask, render_template, request
# import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import torch.nn as nn
from predict import predict_image
import os
from torchvision import transforms
# from pokemon_zukan import classestrans
import requests

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = ['006_Charizard',
           '025_pikachu',
           '026_raichu',
           '133_Eevee',
           '172_pichu',
           '181_Ampharos',
           '393_Piplup',
           '417_pachirisu',
           '587_emolga',
           '658_Greninja',
           '702_dedenne',
           '777_togedemaru',
           '778_mimikyu']

# classes = [
#     '001_フシギダネ',
#     '002_フシギソウ',
#     '003_フシギバナ',
#     '004_ヒトカゲ',
#     '005_リザード',
#     '006_リザードン',
#     '007_ゼニガメ',
#     '008_カメール',
#     '009_カメックス',
#     '010_キャタピー',
#     '011_トランセル',
#     '012_バタフリー',
#     '013_ビードル',
#     '014_コクーン',
#     '015_スピアー',
#     '016_ポッポ',
#     '017_ピジョン',
#     '018_ピジョット',
#     '019_コラッタ',
#     '020_ラッタ',
#     '021_オニスズメ',
#     '022_オニドリル',
#     '023_アーボ',
#     '024_アーボック',
#     '025_ピカチュウ',
#     '026_ライチュウ',
#     '027_サンド',
#     '028_サンドパン',
#     '029_ニドラン♀',
#     '030_ニドリーナ',
#     '031_ニドクイン',
#     '032_ニドラン♂',
#     '033_ニドリーノ',
#     '034_ニドキング',
#     '035_ピッピ',
#     '036_ピクシー',
#     '037_ロコン',
#     '038_キュウコン',
#     '039_プリン',
#     '040_プクリン',
#     '041_ズバット',
#     '042_ゴルバット',
#     '043_ナゾノクサ',
#     '044_クサイハナ',
#     '045_ラフレシア',
#     '046_パラス',
#     '047_パラセクト',
#     '048_コンパン',
#     '049_モルフォン',
#     '050_ディグダ',
#     '051_ダグトリオ',
#     '052_ニャース',
#     '053_ペルシアン',
#     '054_コダック',
#     '055_ゴルダック',
#     '056_マンキー',
#     '057_オコリザル',
#     '058_ガーディ',
#     '059_ウインディ',
#     '060_ニョロモ',
#     '061_ニョロゾ',
#     '062_ニョロボン',
#     '063_ケーシィ',
#     '064_ユンゲラー',
#     '065_フーディン',
#     '066_ワンリキー',
#     '067_ゴーリキー',
#     '068_カイリキー',
#     '069_マダツボミ',
#     '070_ウツドン',
#     '071_ウツボット',
#     '072_メノクラゲ',
#     '073_ドククラゲ',
#     '074_イシツブテ',
#     '075_ゴローン',
#     '076_ゴローニャ',
#     '077_ポニータ',
#     '078_ギャロップ',
#     '079_ヤドン',
#     '080_ヤドラン',
#     '081_コイル',
#     '082_レアコイル',
#     '083_カモネギ',
#     '084_ドードー',
#     '085_ドードリオ',
#     '086_パウワウ',
#     '087_ジュゴン',
#     '088_ベトベター',
#     '089_ベトベトン',
#     '090_シェルダー',
#     '091_パルシェン',
#     '092_ゴース',
#     '093_ゴースト',
#     '094_ゲンガー',
#     '095_イワーク',
#     '096_スリープ',
#     '097_スリーパー',
#     '098_クラブ',
#     '099_キングラー',
#     '100_ビリリダマ',
#     '101_マルマイン',
#     '102_タマタマ',
#     '103_ナッシー',
#     '104_カラカラ',
#     '105_ガラガラ',
#     '106_サワムラー',
#     '107_エビワラー',
#     '108_ベロリンガ',
#     '109_ドガース',
#     '110_マタドガス',
#     '111_サイホーン',
#     '112_サイドン',
#     '113_ラッキー',
#     '114_モンジャラ',
#     '115_ガルーラ',
#     '116_タッツー',
#     '117_シードラ',
#     '118_トサキント',
#     '119_アズマオウ',
#     '120_ヒトデマン',
#     '121_スターミー',
#     '122_バリヤード',
#     '123_ストライク',
#     '124_ルージュラ',
#     '125_エレブー',
#     '126_ブーバー',
#     '127_カイロス',
#     '128_ケンタロス',
#     '129_コイキング',
#     '130_ギャラドス',
#     '131_ラプラス',
#     '132_メタモン',
#     '133_イーブイ',
#     '134_シャワーズ',
#     '135_サンダース',
#     '136_ブースター',
#     '137_ポリゴン',
#     '138_オムナイト',
#     '139_オムスター',
#     '140_カブト',
#     '141_カブトプス',
#     '142_プテラ',
#     '143_カビゴン',
#     '144_フリーザー',
#     '145_サンダー',
#     '146_ファイヤー',
#     '147_ミニリュウ',
#     '148_ハクリュー',
#     '149_カイリュー',
#     '150_ミュウツー',
#     '151_ミュウ'
# ]

# モデル読み込み
# VGG16の場合
# model = models.vgg16_bn(pretrained=False)
# model.classifier[6] = nn.Linear(4096, len(classes))
# MobileNet2の場合
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT) # 事前学習済みモデル
model.classifier[1] = nn.Linear(model.last_channel, len(classes)) # 出力層を調整
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)), # 224×224リサイズ
    transforms.ToTensor(), # テンソル変換
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 平均0.5、標準偏差0.5で正規化
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    label = predict_image(path, model, classes, device="cpu")

    return render_template('result.html', label=label, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)



# ---------------streamlitの場合-----------------
# # アプリのUI
# st.title("ポケモン画像分類アプリ")
# st.write("画像をアップロードしてください")

# # アップロードUI
# uploaded_file = st.file_uploader("画像を選択", type=['jpg', 'jpeg', 'png'])

# # アップロードが行われたら処理
# if uploaded_file is not None:
#     st.image(uploaded_file, caption="アップロードされた画像", use_column_width=True)

#     with st.spinner("分類中..."):
#         result = predict_image(uploaded_file, model, classes, device='cpu')
    
#     st.success(f"このポケモンは「{result}」です！")

# import streamlit as st
# from PIL import Image

# st.title("画像アップロードテスト")

# uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
# st.write("アップロードファイルの内容：", uploaded_file)

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="アップロード画像")
