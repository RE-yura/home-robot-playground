# 3D モデルと素材

配信フォルダに版を付け、以前のキャッシュと混在させません。
部屋のモデル・素材・照明データは `public/assets/realistic-home-v1`、HSR のモデルは `public/assets/hsr-original-v1` から配信します。実行時に外部の素材 API や CDN を呼びません。
家具の装飾と物理の衝突形状を分離し、操作物体・椅子・かご・TV 脇の鉢植えの描画は MuJoCo の body の実際の位置・回転に追従します。

## 公開アセット

以下は **Poly Haven、CC0 1.0** の公開アセットです。商用利用・改変・再配布が可能です。
[配布元のライセンス](https://polyhaven.com/license) / [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)

| 使用箇所 | アセット / 配布元 |
| --- | --- |
| 窓下の収納・TV 台 | [Modern Wooden Cabinet](https://polyhaven.com/a/modern_wooden_cabinet) |
| ラウンジチェア | [Modern Arm Chair 01](https://polyhaven.com/a/modern_arm_chair_01) |
| ソファ前のテーブル | [Modern Coffee Table 01](https://polyhaven.com/a/modern_coffee_table_01) |
| ダイニングチェア | [Dining Chair 02](https://polyhaven.com/a/dining_chair_02) |
| 観葉植物 | [Potted Plant 04](https://polyhaven.com/a/potted_plant_04) |
| 操作できる皿 | [Carved Wooden Plate](https://polyhaven.com/a/carved_wooden_plate) |
| 床の色・凹凸・粗さ | [Laminate Floor 02](https://polyhaven.com/a/laminate_floor_02) |
| 四面の織物調壁紙の色・凹凸・粗さ | [Terlenka](https://polyhaven.com/a/terlenka) |
| 天板・棚などの木材 | [Wood Table 001](https://polyhaven.com/a/wood_table_001) |
| ソファ・ラグの布 | [Fabric Pattern 05](https://polyhaven.com/a/fabric_pattern_05) |
| 表面の反射と環境光 | [Studio Small 09](https://polyhaven.com/a/studio_small_09) |

公開モデルは glTF と 1K の PBR テクスチャを使用。床・木材・布の色は 2K、壁紙と凹凸・粗さの画像は 1K。壁紙は淡いベージュの織物調として Terlenka の写真ベースの色・法線・粗さを使用し、30 cm ごとに繰り返します。高さ 2.7 m の四面の壁で囲み、描画と接触判定は同じ壁の位置・寸法を使用します。天井は設けず、俯瞰カメラは壁の上から室内を見下ろします。
家具は配置先に合わせてサイズを調整し、原点を床面へ揃えます。皿は直径 21 cm・高さ 2.7 cm に合わせ、物理の body 原点へ配置します。
配布ファイルの URL、SHA-256、サイズは `public/assets/realistic-home-v1/sources.json` に記録しています。
既存アセットはダウンロード時に配布元の MD5 も照合しました。追加した Terlenka は配布元の JPEG を保持し、SHA-256 を記録しています。

## このプロジェクトで生成したモデル

Blender でメッシュを生成し、glTF に書き出しています（生成スクリプトは配布していません）。
机、収納かご、花柄のマグ、ボトル、缶、青い紙箱、テニスボール、野球ボール、ソファ、冷蔵庫、キッチン、テレビ、本棚、お茶・ジュース・牛乳の16モデルです。
中央のローテーブルは天板高さ 40 cm・厚さ 4 cm、短い金属脚と木材の PBR 素材を持ち、描画と衝突形状を同じ寸法に揃えます。
マグは内側・底・取っ手の穴、ボトルは肩・首・キャップの溝・ラベル、家具・家電は丸い縁・継ぎ目・取っ手を持ちます。
これは生成 AI によるスキャン再構成ではなく、寸法を指定した Blender でのモデリングです。

缶・ボトルの全面ラベル、マグの花柄、箱のラベル、ボールの表面画像はオリジナルの図案です。`materials/printed` に同梱します。缶・ボトルは円周方向の UV で表面へ印刷し、浮いた立体文字を置きません。かごの四面の途切れない格子と左右の側面に付く横棒の取手はメッシュと `src/simulation/basket-shapes.json` の当たり判定を同時に生成します。縫い目と格子は同じ素材ごとに一つのメッシュへまとめています。

テニスボールは直径67 mm・58 g、野球ボールは直径73 mm・145 gの球として接触を計算します。寸法・重量の参考は [ITF の技術資料](https://www.itftennis.com/media/15639/2026-technical-booklet.pdf) と [MLB の公式規則](https://content.mlb.com/documents/2/2/4/305750224/2019_Official_Baseball_Rules_FINAL_.pdf) です。空気圧・布や革の変形・規格どおりの跳ね返りは再現していません。

画像は元の PBR 素材を共有するため、モデルごとに重複して埋め込みません。
ロボット全体のメッシュは下記の HSR 公開モデルを使用し、MuJoCo の実測リンク姿勢に追従させます。

## HSR の公式公開モデル

[Toyota Motor Corporation の HSR mesh（TRI による公開）](https://github.com/ToyotaResearchInstitute/hsr_meshes) の台車・胴体・頭・カメラ・肩・腕・手首・掌・左右の指を使用します。
ライセンスは **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**。非商用用途に限られ、改変した素材の共有は認められません。この個人用 Site では元の OBJ・MTL・テクスチャをバイト単位で保持し、原寸で座標系と各リンクの表示位置を変換しています。メッシュの編集・間引き・再着色は行っていません。
出典・上流 commit・ファイルの SHA-256 は `public/assets/hsr-original-v1/sources.json`、原文ライセンス・保証免責は同フォルダの `LICENSE.txt`、画面から辿れるクレジットは `ATTRIBUTION.html` に収録します。
一部の OBJ は面と線を同じオブジェクトに含みます。Three.js の OBJLoader が面まで線として描画する問題を避けるため、表示時には元の面・法線・UV を読み、線プリミティブを描画しません。配信する元ファイルのバイト列は変更していません。
表示時の材質は元の MTL の色・テクスチャ・透明度を保持して PBR に変換し、家具と同じ HDR 環境光を受けるようにしています。腕・手首の白い外装は非金属・粗さ 0.30 と薄いクリアコート、手首の金属部は金属度 0.75・粗さ 0.32、指は粗さ 0.78 の艶を抑えた表面として描画します。これらは表示用の仮の材質値です。
腕・手首には面と逆向きの法線が含まれるため、表示時だけ面から法線を再計算します。60°以上の角を残し、10 µm の位置許容差で曲面を滑らかにします。元の頂点位置・UV・面数と配信ファイルは変更せず、形状と物理判定を保持します。

[HSR の URDF description](https://github.com/ToyotaResearchInstitute/hsr_description) は BSD-3-Clause-Clear で、メッシュとは別のライセンスです。原文は `DESCRIPTION-LICENSE.txt`、元 URDF は `hsrb4s.obj.urdf` に保持します。`tools/import-hsr.py` が各関節の原点・軸・可動域・慣性・visual / collision と指の mimic を抽出し、`src/robot/hsr-description.json` に保存します。衝突メッシュは元の頂点を MuJoCo の凸包として使用します。台車の衝突・移動は従来の近似モデルで、腕・頭・指の制御ゲインや接触設定も実験用です。HSR の制御ソフトは使用していません。Toyota の公式製品・提携・推奨を意味しません。

## 見た目と物理

動くモデルの外寸と body 原点を MuJoCo の衝突形状に照合するテストがあります。マグなどの曲面は、既存の複数の単純な衝突形状で近似します。
食卓の椅子は公開 glTF の三角形を座面・４本の脚・背もたれ４段へ分割し、各部分の凸包を使用します。背もたれを一つの箱や凸包にせず、曲線の外側に空いた隙間を塞ぐことを防ぎます。描画と同じ床面・中央原点に揃え、体積に比例して 6 kg の質量を分配します。生成済みの `src/simulation/dining-chair-collision.json` を使用し、通常のビルドで再生成は不要です。

TV 脇の鉢植えは Potted Plant 04 を幅 19 cm・高さ 29 cm・奥行き 20 cm に合わせ、鉢の表面から凸包を生成しています。`src/simulation/tv-plant-collision.json` が描画寸法と接触形状を共有します。通常のビルドで再生成は不要です。鉢と土を合わせた仮の質量は 1 kg、滑り摩擦係数は 0.55。葉は鉢と一緒に動きますが、葉ごとの接触と変形は省略しています。窓下の鉢植えは装飾のままです。

装飾のラベル・溝・縫い目には個別の衝突判定を設けません。壁・窓と家具・家電にも複合衝突形状があります。２種類の椅子・かご・TV 脇の鉢植えは移動できる自由剛体で、家具・家電の本体は固定です。冷蔵庫の下側の扉は、6 kg の受動的な蝶番で 0〜100° の範囲を動き、取手を指の摩擦で掴んで引くと開きます。描画は実測の関節角に追従し、扉・パッキン・取手が一緒に動きます。閉じ際の約 4.6° は最大 1.4 N·m の吸着トルクでシールを近似します。実機の測定値ではなく操作用の仮定です。庫内は空洞で、側板・背板・底・棚板に個別の接触判定があります。

キッチンの左収納扉は2 kg・0〜105°の横開き、右のオーブン扉は2.5 kg・0〜90°の下開きです。扉・取手を独立したglTFノードとして保ち、MuJoCoの実測関節角に追従させます。オーブンの閉じるばねは4.5 N·m/rad、蝶番の減衰・摩擦・吸着も実験用の仮値です。庫内の3種類の飲みものは専用の形とオリジナルの印刷ラベルを持ち、胴・肩・首・キャップを複合接触形状で近似します。重量はお茶550 g、ジュース420 g、牛乳360 gです。

配置は描画と物理で共有しています。椅子の脚や棚板などを個別の形状にして、空間を残します。取っ手・椅子の細い傾斜部などは近似で、布の変形、冷蔵庫の上側の扉と他の収納の開閉は模擬しません。植物の葉やラグの細部は装飾です。

ブラウザのリアルタイム表示を優先した PBR と影・環境反射です。パストレーシング、完全な間接光、液体、破損、実機 HSR の精密動力学は含みません。
