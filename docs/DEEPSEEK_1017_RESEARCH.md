# DeepSeek 1017を小容量RAMで動かす方法の調査

調査日：2026年10月17日。対象はRTX 3090・32GB RAM級の単一GPU環境です。論文の記述、公開実装、1017 Q2_Kのヘッダーを確認しました。ビルド・変換・推論・ベンチマークは実施していません。以下の優先順位と適用判断は、この環境に対する推論です。

## まず減らすべき転送量

取得対象のQ2_Kは、gate/upがQ2_K、downがQ3_Kです。確認したテンソル形状から計算すると、expert一つは次の大きさです。

```text
5120 × 2304 × (84 + 84 + 110) / 256 = 12,810,240 bytes
40層 × top-6 × 12,810,240 = 3,074,457,600 bytes/token
```

約3.07GB/トークンは、選択されたexpertがすべてRAM/VRAMを外した場合の論理読み出し量です。実測速度ではありません。expert全体は196.77GBあり、32GB RAMに全部は載りません。限られたRAMには、繰り返し使われるexpertを残す価値があります。

一方、内蔵Q2_K Engramは最大48行×84バイト＝4,032バイト/トークンです。こちらは帯域より、ランダムアクセスの回数・遅延と行キャッシュが課題になります。物理I/Oはページ単位の増幅を含むため、4,032バイトだけSSDを読めばよいという意味ではありません。n-gramの表全体をRAMへ移すことと、行キャッシュを持つことは分けて考えます。

数値の根拠は、このForkの固定Q2_Kマニフェストと全1,046テンソルのヘッダーです。実装と入力契約は [DEEPSEEK_1017_Q2_K.md](DEEPSEEK_1017_Q2_K.md) に記録しています。

## 方法の比較

同じ「高速化」でも、重みの転送を減らす方法、転送を隠す方法、生成回数を減らす方法があります。

| 優先度 | 論文・方式 | 使えそうな部分 | この環境での制約 |
|---|---|---|---|
| 高 | [SSD-LLaMA](https://arxiv.org/html/2609.18110v1) | SSD/RAM/VRAMの固定容量キャッシュ、expert単位の配置、読み出し完了ごとの非同期転送 | 主な評価機はRTX 5090・PCIe 5.0。1017/3090の速度証拠にはならない |
| 高 | [FlashMoE：SSDキャッシュ置換](https://arxiv.org/html/2601.17063v2) | 使用頻度と直近の再利用を合わせ、SSDからの読み直しを減らす | 学習型置換にはrouting traceと学習が必要。まずLRU/LFUとの比較が必要 |
| 高 | [MoE-Infinity](https://arxiv.org/abs/2401.14361) | リクエスト単位のアクセス履歴を使ったキャッシュ・先読み | 論文の効果と公開ライブラリの実装・対応モデルは別に確認する |
| 高〜中 | [Fiddler](https://arxiv.org/abs/2402.07033) | RAM内expertはCPUで計算し、重みのPCIe転送を避ける | SSDにしかないexpertの読み出しは残る。CPU帯域とGPU待ちの実測が必要 |
| 中 | [MoE-SpeQ](https://arxiv.org/html/2511.14102v1) | ドラフトを次トークン予測だけでなくexpert先読みに使う。I/Oに応じてdraft長を調整 | ドラフト用VRAM、誤った先読み、追加expert読み込みが必要になる |
| 中〜低 | [DraftExpert](https://arxiv.org/html/2607.24434v1) | 採択トークン数と、新たに必要になるexpert数を一緒に最適化 | 専用draft expertの蒸留が必要。既製1017に設定一つで適用する方式ではない |
| 中 | [LLM in a flash](https://arxiv.org/abs/2312.11514) | 再利用する重みを残すwindowing、連続読み出しに寄せるrow-column bundling | 元の主対象と1017 MoEの構成は異なる。配置・I/O設計の参考 |
| 中 | [Lookahead Decoding](https://arxiv.org/abs/2402.02057) | 別のdraftモデルを持たない並列検証 | 検証する候補が増えるとMoEのexpert集合も増え、SSD読み出しが悪化し得る |
| 中〜低 | [HOBBIT](https://arxiv.org/abs/2411.01433) | 重要度に応じたexpert精度と読み込み量の調整 | キャッシュミス時の低精度版への置換は数値を変える。既存Q2_Kはすでに低精度 |
| 条件付き | [Engram原論文](https://arxiv.org/html/2601.07372v1) | 入力から行番号が決まるため、行先読みと計算を重ねられる | 原論文のオフロード実験はホストDRAM。SSDで同じ低下率とは言えない |
| 長文時 | [KIVI](https://arxiv.org/abs/2402.02750) | KVの低ビット化。key/valueで量子化の軸を変える | 1017の共有・圧縮KVとindexerへの移植・数値検証が必要 |
| 長文時 | [STAR-KV](https://arxiv.org/html/2606.08382v1) | 低ランク化と量子化でKVを縮小し、空いたVRAMをexpertに回す案 | 校正・学習と専用カーネルが必要。既存の1017 attentionに直接対応した証拠は未確認 |
| 用途限定 | [SSR](https://arxiv.org/abs/2610.01892) | 固定的なエージェント手順を候補選択にして生成量を減らす | 2B/4Bのマルチモーダル検索タスクでの評価。一般チャットやコード推論全体の置換にはならない |

FlashMoEは同名プロジェクトが複数あります。この表はKAIST等のSSDキャッシュ置換論文 `2601.17063` を指します。分散MoEの単一カーネル実装や、別のfine-grained MoEライブラリとは混ぜていません。

## 実装へ取り込む順番

第一候補は、expertのRAM/VRAM常駐とSSD読み出しの改善です。SSD-LLaMAはexpertを連続領域へまとめ、並列direct I/Oと転送を重ね、必要ならrANSで無損失圧縮します。ただしその圧縮対象は主に4ビット形式です。現在のQ2_K/Q3_Kでも同じ圧縮率になるとは言えず、追加packを作る容量も必要です。今の直接読み込み経路を基準にし、再配置の有無だけを比較する順が妥当です。[論文 §4.1・§5.1](https://arxiv.org/html/2609.18110v1)

キャッシュは「RAMを全部埋める」ことより、「同じ容量でSSD読み直しを何バイト減らせるか」で評価します。FlashMoEの発想は近いですが、同論文のLRU比較での速度改善はモデルにより約7%と22%です。要旨の最大2.6倍を、キャッシュ置換だけの効果として使わないようにします。Xolotlにはすでにarena・VRAM LRU・先読みがあるため、既存機構との重複を先に確認します。[FlashMoE §4.3–4.4](https://arxiv.org/html/2601.17063v2)

CPU計算とGPU転送の選択は、RAMに置いた場所で決め打ちしない方がよさそうです。FiddlerはCPU計算を使って転送を減らします。一方、SSD-LLaMAの評価機ではGPUへ寄せた構成が有利でした。3090環境でも、RAMヒット時のCPU処理時間とH2D＋GPU処理時間を後で比較する必要があります。[Fiddler](https://arxiv.org/abs/2402.07033)、[SSD-LLaMA §4.3](https://arxiv.org/html/2609.18110v1)

投機的生成はその次です。MoE-SpeQとDraftExpertに共通する注意は、複数候補の検証で新しいexpertが増えると、採択率が高くても転送が増え得る点です。採択率だけでなく、採択トークン当たりのSSDバイト数・追加expert数・検証時間を残します。既存の1017対応DSparkを準備できるかも比較対象ですが、旧重みを流用しません。[MoE-SpeQ §3.2](https://arxiv.org/html/2511.14102v1)、[DraftExpert](https://arxiv.org/html/2607.24434v1)

## 先に挙がっていた数字の読み方

- Engram論文の最大2.8%の低下は、H800、512系列、ホストDRAMに100Bパラメータの表を置いた測定です。3090・単一会話・SSDの数字ではありません。[原論文 Table 4](https://arxiv.org/html/2601.07372v1)
- STAR-KVの最大6.9倍はattention部分の値で、64K文脈の比較対象にメモリ不足による推定を含みます。学習には校正データも使います。1017全体を設定一つで6.9倍にする根拠にはできません。[原論文 §5・§7](https://arxiv.org/html/2606.08382v1)
- SSRの推論レイテンシ削減は、事前候補を用意した検索エージェントでの結果です。自由な会話・一般的な長い思考にも同じ割合が出るという意味ではありません。[原論文](https://arxiv.org/abs/2610.01892)
- NVFP4のネイティブTensor Core経路はBlackwell向けです。3090で低ビット重みを扱えることと、その専用演算器で加速されることは分けます。量子化方式名だけで2〜3倍とは置きません。[NVIDIAの対応表](https://docs.nvidia.com/nim/large-language-models/latest/supported-architectures.html)

## エンジンを変える案

別候補として、ColibriにはV4.1 Flashを元のFP8/FP4 safetensorsから直接読み、EngramをSSDから行単位で読む実装があります。型を保った比較対象になる可能性があります。ただし公開ドキュメントで1017明示と3090実測は確認できていません。また入力は約510GBの公式重みで、現在の約265GB Q2_K取得とは別です。追加ダウンロードや切り替えは実施していません。[V4.1資料](https://github.com/JustVugg/colibri/blob/main/docs/deepseek-v41.md)、[Vulkan対応表](https://github.com/JustVugg/colibri/blob/main/docs/vulkan.md)

MoE-Infinityの公開版はSSD/RAM/VRAMと推論サーバーを扱いますが、作者自身が論文版と公開版の違いを説明しています。機構の参考と、1017をそのまま動かせる実装かどうかは別の評価です。[公式README](https://github.com/EfficientMoE/MoE-Infinity)

単一GPUでは、まずメモリ予算内の一つのプロセスを維持し、会話のprefixやKVを再利用する方が取り組みやすいと判断します。大規模クラスタ向けのprefill/decode分離やcontinuous batchingを、そのまま優先策にはしません。

## 後で測るもの

Comfy優先の解除後に、ビルドと数値一致を確認してから測ります。今は実行しません。

1. routing bias・expert省略を無効にした1017基準。形式対応と速度改善を混ぜない。
2. prefillとdecodeを別々に測る。TTFT、トークン間の遅延、SSDバイト/採択トークン、expertのRAM/VRAMヒット率を記録。
3. RAM/VRAMの予算を固定し、OS・Comfy・KV・pinned buffer・ページキャッシュの余裕を確保。
4. expert cacheとEngram行cacheの予算配分を比較。旧モデルのrankを1017の最適配置とみなさない。
5. exact routingを維持するキャッシュ・先読み・配置から比較し、精度を変える再量子化・expert削減は別実験にする。

記事化する際も、論文中の結果、この環境への適用仮説、このForkのソース実装、実機で得た結果を区別します。この時点で存在するのは最初の三つまでです。
