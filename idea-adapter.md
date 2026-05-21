# 清書案：AnyAdapter-style Disturbance Adapter for Recovery-Gated AMP Locomotion

## 0. 一言でいう案

**AMPで作った通常歩行policyをbaseとして固定し、その上にhistory-based adapterを追加して、push・摩擦変化・質量変化・latencyなどへの補正だけを学習する。**

つまり、policyを一枚で全部fine-tuneしない。

[
\text{human-like nominal gait}
\quad+\quad
\text{disturbance adaptation}
]

を構造的に分ける。

Any2Trackの重要な思想は、基本動作能力と外乱適応能力を分離することです。論文では、まず外乱なしでbase trackerを学習し、その後にhistory-informed AnyAdapterを追加して、terrain、external force、physical property changeに適応させています。さらに、base trackerをfreezeしてadapterだけをfine-tuneすることで、既に獲得したmotion tracking能力を壊さない設計になっています。([arXiv][1])

君の問題では、これを **motion tracking policy** ではなく **AMP-based locomotion policy** に移植する。

---

# 1. 提案名

仮称：

**Recovery-Gated AMP with Frozen Dynamics Adapter**

または短く、

**AMP-Adapter**

論文タイトルっぽくするなら、

> **Preserving Human-like Locomotion under Push Disturbances via Frozen AMP Base and History-informed Dynamics Adapter**

---

# 2. 中核仮説

今までの問題はこれ。

[
r =
r_{\text{task}}
+
\lambda_{\text{AMP}}r_{\text{AMP}}
+
r_{\text{recovery}}
]

として一枚policyを学習すると、push recoveryを入れた瞬間にAMPで作った人間らしい歩きが崩れる。

原因は、policyが同時に

1. normal walkingらしく動け
2. 外乱時は大きく踏み替えろ
3. torqueを抑えろ
4. velocity commandも追え

という矛盾した要求を受けるから。

AnyAdapter-styleにすると、この矛盾を少し分解できる。

[
a_t =
a^0_t + \Delta a_t
]

ここで、

[
a^0_t = \pi_{\text{AMP-base}}(o_t, c_t)
]

は人間らしい通常歩行を担当する固定base policy。

[
\Delta a_t =
\xi(o_t, c_t, e_t)
]

は外乱・物理差分・push recoveryのためのadapter補正。

つまり、

> **base policyは歩き方を決める。adapterは崩れたときの補正を決める。**

---

# 3. 全体アーキテクチャ

## 3.1 Frozen AMP base policy

まずAMPで通常歩行policyを作る。

[
a^0_t =
\pi_{\theta}^{\text{base}}(o_t, c_t)
]

入力：

[
o_t =
[
q_t,
\dot q_t,
a_{t-1},
\omega_{\text{base}},
g_{\text{proj}},
h_{\text{base}},
\text{foot contacts}
]
]

command：

[
c_t =
[
v^{cmd}_x,
v^{cmd}_y,
\omega^{cmd}_z
]
]

出力：

[
a^0_t
]

これはPD target、またはPD target offset。

このbase policyは、normal walking、turning、speed variationをAMPで学習する。
ここで強いpushを入れすぎない。目的はrobustnessではなく、**人間らしいnominal gaitを作ること**。

学習後、

[
\theta^{\text{base}} \leftarrow \text{frozen}
]

にする。

ここが甘いと失敗する。baseをfreezeしないと、adapter training中に人間らしい歩きが普通に壊れる。

---

## 3.2 History encoder

外乱や物理パラメータは直接観測できない。
だから、最近の状態・行動履歴から推定する。

[
h_t =
{
o_{t-H}, a_{t-H},
\dots,
o_{t-1}, a_{t-1}
}
]

[
e_t =
\phi_{\psi}(h_t)
]

ここで (e_t) はdynamics embedding。

これは明示的に「push force」や「摩擦係数」を教師ありで当てる必要はない。
履歴から、現在の環境・外乱・ロボット状態をpolicyが使えるlatentに圧縮する。

Any2Trackでは、history encoderからdynamics embeddingを作り、それをworld model predictionのproxy taskで学習しています。論文中の設定ではhistory window (H=79)、autoregressive prediction window (N=20) が使われていますが、この値を盲目的にコピーする必要はありません。([arXiv][1])

実装ではまず、

[
H = 10 \sim 40
]

くらいから始める方が現実的。
制御周期が50 Hzなら、0.2〜0.8秒程度の履歴。push recoveryにはこの範囲がまず効く。

---

## 3.3 World model auxiliary loss

history encoderをまともにするために、world model lossを入れる。

[
\hat{s}_{t+i+1}
===============

\omega_{\eta}(\hat{s}*{t+i}, a*{t+i}, e_t)
]

[
L_{\text{wm}}
=============

\sum_{i=1}^{N}
|
\psi_s(s_{t+i}) -
\psi_s(\hat{s}_{t+i})
|_1
]

ここで (\psi_s(\cdot)) は予測対象の状態特徴。

予測する候補：

[
\psi_s(s_t)
===========

[
q_t,
\dot q_t,
\omega_{\text{base}},
g_{\text{proj}},
h_{\text{base}},
v_{\text{base}},
\text{foot contacts}
]
]

注意点は、world modelを重くしすぎないこと。
目的は高精度な物理シミュレータを作ることではない。
**adapterに使えるdynamics embeddingを作ること**。

Any2Trackでも、history情報をそのまま使うとノイズや不要情報が多いため、world model predictionをproxy taskとしてdynamics featureを抽出する設計になっています。([arXiv][1])

---

## 3.4 Adapter

adapterはbase policyの出力を補正する。

最小実装はaction residual。

[
\Delta a_t =
\xi_{\zeta}(o_t, c_t, e_t)
]

[
a_t =
a^0_t + \Delta a_t
]

よりAnyAdapterに近い実装はlayer-wise adapter。

base networkの各層を、

[
z^{l+1} = f^l(z^l)
]

として、adapterを足す。

[
z^{l+1} = f^l(z^l) + \xi^l(z^l, e_t)
]

最初はaction residual版でいい。
論文実装や主張を強くしたい段階でlayer-wise adapterにすればいい。

adapterはzero initializationにする。

[
\xi_{\zeta}(\cdot) \approx 0
]

したがって、training開始時は

[
a_t \approx a^0_t
]

になる。

これが重要。
最初からadapterが大きく出ると、AMP baseを壊す。Any2Trackでもzero-initialized adapterを使い、fine-tuning開始時にはbase trackerの出力を変えず、学習が進むにつれて外乱適応を注入する設計になっています。([arXiv][1])

---

# 4. Recovery-gated AMP reward

adapterを入れても、AMPを常時効かせるとrecoveryを邪魔する。
だからAMP rewardは状態依存でgateする。

まずdisturbance scoreを作る。

[
\delta_t =
\alpha_1 |\theta_{\text{roll,pitch}}|
+
\alpha_2 |\omega_{xy}|
+
\alpha_3 |v_{xy} - v^{cmd}*{xy}|
+
\alpha_4 |h - h_0|
+
\alpha_5 e*{\text{capture}}
+
\alpha_6 e_{\text{contact}}
]

style gate：

[
g_t =
\exp(-\beta \delta_t)
]

安定時：

[
g_t \approx 1
]

外乱時：

[
g_t \approx 0
]

最終reward：

[
r_t =
r^{task}_t
+
g_t r^{AMP}_t
+
(1-g_t) r^{rec}_t
+
r^{reg}_t
---------

\lambda_{\Delta a}(g_t)|\Delta a_t|^2
]

ここがこの案の核。

* 安定時：AMPを効かせる
* 外乱時：AMPを弱める
* recovery時：普通のPPO rewardで踏み替え・姿勢回復を学習する
* adapter：必要なときだけbase actionからズレる

---

# 5. Adapter residual penalty

adapterが常にbaseを上書きすると、結局AMP baseを壊す。
だからresidual penaltyを入れる。

[
r_{\Delta a}
============

*

\lambda_{\Delta a}(g_t)
|\Delta a_t|^2
]

状態依存にする。

[
\lambda_{\Delta a}(g_t)
=======================

\lambda_{\text{stable}} g_t
+
\lambda_{\text{rec}}(1-g_t)
]

[
\lambda_{\text{stable}} > \lambda_{\text{rec}}
]

意味はこう。

安定時：

[
g_t \approx 1
\Rightarrow
\lambda_{\Delta a} \approx \lambda_{\text{stable}}
]

つまり、baseからズレるな。

外乱時：

[
g_t \approx 0
\Rightarrow
\lambda_{\Delta a} \approx \lambda_{\text{rec}}
]

つまり、必要なら大きく補正していい。

これを入れないと、adapterが「常時robust but ugly」なpolicyに変質する。
その瞬間、この案の価値は消える。

---

# 6. Recovery reward

recoveryはmocap imitationではなく、reward-based PPOで学習させる。

[
r^{rec}*t =
r*{\text{upright}}
+
r_{\text{height}}
+
r_{\text{angvel-damp}}
+
r_{\text{no-fall}}
+
r_{\text{foot-placement}}
+
r_{\text{return}}
]

## Upright

[
r_{\text{upright}}
==================

\exp(-|\theta_{\text{roll,pitch}}|^2/\sigma_{\theta})
]

## Base height

[
r_{\text{height}}
=================

\exp(-(h-h_0)^2/\sigma_h)
]

## Angular velocity damping

[
r_{\text{angvel-damp}}
======================

\exp(-|\omega_{xy}|^2/\sigma_{\omega})
]

## No fall

[
r_{\text{no-fall}}
==================

\mathbb{1}[\text{not fallen}]
]

## Foot placement

capture pointやDCMを見て、足を出す方向を誘導する。

[
r_{\text{foot-placement}}
=========================

\exp(-e_{\text{capture-foot}}^2/\sigma_f)
]

ただし、足位置を強く固定しすぎない。
recoveryでは普通の歩行よりも大股・横ステップ・後ろステップが必要になる。

## Return-to-command

安定後にcommand trackingへ戻す。

[
r_{\text{return}}
=================

\mathbb{1}[\delta_t < \delta_{\text{stable}}]
\exp(-|v_{xy} - v^{cmd}_{xy}|^2/\sigma_v)
]

---

# 7. Task reward

command trackingは必要だが、push直後に強制しすぎると倒れながら速度だけ追う挙動が出る。

[
r^{task}*t =
w*{\text{cmd}}(\delta_t)
[
\exp(-|v_{xy} - v^{cmd}_{xy}|^2/\sigma_v)
+
\exp(-|\omega_z - \omega^{cmd}*z|^2/\sigma*{\omega})
]
]

[
w_{\text{cmd}}(\delta_t)
========================

w_{\min} + (1-w_{\min})g_t
]

安定時はcommand trackingを強く、外乱時は少し緩める。

---

# 8. AMP discriminatorの扱い

adapter training中にAMP discriminatorを雑に更新すると危険。
push recovery中の崩れた姿勢をdiscriminatorが大量に見ると、style rewardの意味がブレる。

おすすめはこれ。

## Option A: AMP discriminatorをfreeze

Stage 1でAMP baseを学習したあと、discriminatorも固定する。

これが最も安全。

## Option B: stable segmentだけで更新

[
g_t > g_{\text{stable}}
]

を満たす区間だけをpolicy sampleとしてdiscriminator更新に使う。

外乱中・recovery中の動きをnormal walking discriminatorに食わせない。

最初はOption Aでいい。
ここで複雑にするとdebugが地獄になる。

---

# 9. Training schedule

## Stage 0: Robust PPO baseline

AMPなしで、普通のlocomotion reward + push curriculumを学習する。

目的：

* そのロボットがそもそもrewardだけでpush recoveryできるか確認する
* recovery rewardをdebugする
* max recoverable impulseの下限を知る

ここで弱いなら、adapter以前の問題。

---

## Stage 1: Train AMP base policy

AMPで通常歩行を作る。

[
\pi^{\text{base}}_{\theta}
]

条件：

* normal walking
* turning
* speed variation
* mild terrain / mild dynamics randomization程度
* 強いpushは入れない

目的：

* human-like nominal gait
* natural arm swing
* natural contact rhythm
* stable command tracking

この段階で、人間らしさを最大化する。

---

## Stage 2: Freeze base

[
\theta^{\text{base}} \leftarrow \text{frozen}
]

さらにAMP discriminatorも基本的にはfreeze。

この段階で守るべき資産は、

> AMPで得たnominal human-like gait

です。

---

## Stage 3: Pretrain history encoder + world model

frozen baseを使って、外乱ありのrolloutを集める。

外乱：

* external push
* friction randomization
* torso mass randomization
* CoM shift
* motor strength variation
* latency
* sensor noise
* light terrain variation

履歴からdynamics embeddingを作り、world model lossでpretrainする。

[
L_{\text{wm}}
=============

\sum_i
|
\psi_s(s_{t+i}) -
\psi_s(\hat{s}_{t+i})
|_1
]

ここでpolicyはまだ大きく変えない。
先に「外乱を履歴から読む能力」を作る。

---

## Stage 4: Train adapter with PPO

baseはfreezeしたまま、adapterだけをPPOで学習する。

[
a_t =
\pi^{\text{base}}(o_t,c_t)
+
\xi(o_t,c_t,e_t)
]

reward：

[
r_t =
r^{task}_t
+
g_t r^{AMP}_t
+
(1-g_t) r^{rec}_t
+
r^{reg}_t
---------

\lambda_{\Delta a}(g_t)|\Delta a_t|^2
]

ここでpush curriculumを入れる。

最初は弱いpush。
徐々にimpulseを上げる。

---

## Stage 5: Joint update of adapter and encoder

adapterとhistory encoderを交互、または同時に更新する。

実装上はこうでいい。

1. rollout collection
2. world model update
3. PPO update for adapter
4. repeat

Any2Trackもadapterとhistory encoderを交互に更新する形を取っています。([arXiv][1])

---

## Stage 6: Failure-aware push mining

random pushだけでは足りない。

失敗条件を記録する。

[
(
\text{gait phase},
\text{push direction},
\text{push magnitude},
\text{push duration},
\text{push location},
\text{command}
)
]

失敗しやすい条件を高確率で再サンプルする。

特に見るべき条件：

* lateral push during single support
* backward push during forward walk
* diagonal push during turning
* shoulder push causing yaw moment
* low-friction lateral shove

---

# 10. Push curriculum

外乱強度はforceではなくimpulseで管理する。

[
I = F\Delta t
]

さらにmass normalizeする。

[
\Delta v = I/m
]

curriculum例：

## Level 0

pushなし。
AMP base確認。

## Level 1

小さいpush。

[
\Delta v = 0.1 \sim 0.3 \text{ m/s}
]

## Level 2

中程度push。

[
\Delta v = 0.3 \sim 0.7 \text{ m/s}
]

## Level 3

強push。

[
\Delta v = 0.7 \sim 1.2 \text{ m/s}
]

## Level 4

phase-aware / failure-aware push。

direction、timing、locationを失敗ケースに寄せる。

---

# 11. 実装の最小構成

最初から全部やるな。
まずこの最小構成でいい。

[
a_t =
a^0_t + \Delta a_t
]

[
e_t =
\text{GRU}(o_{t-H:t}, a_{t-H:t-1})
]

[
\Delta a_t =
\text{MLP}(o_t, c_t, e_t)
]

[
r_t =
r^{task}_t
+
g_t r^{AMP}_t
+
(1-g_t)r^{rec}_t
+
r^{reg}_t
---------

\lambda_{\Delta a}(g_t)|\Delta a_t|^2
]

最初はlayer-wise adapterまでやらなくていい。
action residual adapterで効果が出ないなら、layer-wiseにしても救えない可能性が高い。まず設計思想が効くかを見る。

---

# 12. Ablation

最低限、これを比較する。

## A. No adapter

[
\pi = \pi_{\text{AMP}}
]

constant AMP + push。

## B. Gated AMP only

[
r =
r^{task}
+
g r^{AMP}
+
(1-g)r^{rec}
]

adapterなし。

## C. Frozen base + adapter without world model

history encoderはPPO lossだけで学習。

## D. Frozen base + adapter + world model

本命。

## E. Non-frozen fine-tuning

baseもadapterも全部更新。

これはたぶんstyleが崩れる。
崩れたら主張が強くなる。

## F. Full model

[
\text{frozen AMP base}
+
\text{history encoder}
+
\text{world model}
+
\text{adapter}
+
\text{gated AMP}
+
\text{failure-aware push}
]

---

# 13. 評価指標

Any2Track系のMPJPEだけでは足りない。
君の問題はpush-recoveryなので、評価はpush中心にする。

## Robustness

* fall rate
* max recoverable impulse
* recovery time
* consecutive push recovery rate
* fall rate vs impulse magnitude
* fall rate vs gait phase
* fall rate vs push direction

## Human-likeness

* AMP score before push
* AMP score after recovery
* gait frequency
* arm swing symmetry
* contact rhythm
* step width / step length distribution

## Command tracking

* post-push velocity tracking error
* command recovery time
* yaw tracking error
* lateral tracking error

## Safety

* peak torque
* torque rate
* foot slip
* impact force
* joint limit violation
* non-foot contact rate
* self-collision

## Adapter behavior

これは必ず見る。

[
|\Delta a_t|
]

を時間でplotする。

理想はこう。

* 安定歩行中：(|\Delta a_t|) が小さい
* push直後：(|\Delta a_t|) が大きくなる
* recovery後：また小さくなる

もし安定時もadapterが大きく出ているなら、baseが壊されている。
その場合、この設計は失敗。

---

# 14. 失敗パターンと対策

## 失敗1: Adapterが常にbaseを上書きする

症状：

* nominal gaitがロボット臭くなる
* AMP scoreが下がる
* (|\Delta a_t|) が常時大きい

対策：

[
\lambda_{\text{stable}}
]

を上げる。
または、stable時のadapter outputに強いpenaltyを入れる。

---

## 失敗2: Push時にadapterが弱すぎる

症状：

* 人間らしさは保つが、pushで転ぶ
* (|\Delta a_t|) がpush時にも小さい

対策：

[
\lambda_{\text{rec}}
]

を下げる。
recovery rewardを強くする。
push curriculumをゆっくりにする。

---

## 失敗3: GateがすぐAMPを切る

症状：

* 少しの速度誤差でAMP rewardが消える
* normal walkingが不安定になる

対策：

[
\delta_t
]

のscaleをstable rolloutのpercentileで正規化する。

例えば、

[
\tilde{\delta}*t =
\delta_t / P*{95}(\delta_{\text{stable}})
]

としてからgateに入れる。

---

## 失敗4: World modelが役に立たない

症状：

* w/o world modelと差が出ない
* (e_t) が外乱条件を分離していない

対策：

* 予測対象を絞る
* base velocity、projected gravity、foot contact、heightを重視する
* 長すぎるhorizon予測をやめる
* (N=1\sim5) から始める
* latent dimensionを小さくしすぎない

---

## 失敗5: Frozen baseが弱すぎる

症状：

* adapterが常に大補正しないと歩けない
* nominalでも(\Delta a_t)が大きい

対策：

Stage 1のAMP baseを作り直す。
ここをごまかすな。baseが弱いとadapterは補正器ではなく、第二のpolicyになってしまう。

---

# 15. この案の主張

研究としての主張はこう。

> AMP-based humanoid locomotion can produce human-like nominal gait, but directly training the same policy for strong push recovery often degrades the learned style. We address this by freezing the AMP base policy and learning a history-informed dynamics adapter that injects disturbance-specific action corrections. A disturbance-dependent style gate suppresses the nominal AMP reward during recovery, allowing reward-based push recovery without sacrificing nominal human-likeness.

もっと短くすると、

> **AMP learns how to walk naturally. The adapter learns how to survive disturbances. The gate decides when style should yield to recovery.**

---

# 16. 最終構成

最終policy：

[
a_t =
\pi_{\text{AMP-base}}(o_t,c_t)
+
\xi(o_t,c_t,\phi(h_t))
]

baseはfreeze。

[
\theta_{\text{base}} = \text{fixed}
]

history embedding：

[
e_t = \phi(h_t)
]

reward：

[
r_t =
r^{task}_t
+
g_t r^{AMP}_t
+
(1-g_t) r^{rec}_t
+
r^{reg}_t
---------

\lambda_{\Delta a}(g_t)|\Delta a_t|^2
]

gate：

[
g_t =
\exp(-\beta \delta_t)
]

disturbance score：

[
\delta_t =
\alpha_1 |\theta_{\text{roll,pitch}}|
+
\alpha_2 |\omega_{xy}|
+
\alpha_3 |v_{xy} - v^{cmd}*{xy}|
+
\alpha_4 |h - h_0|
+
\alpha_5 e*{\text{capture}}
+
\alpha_6 e_{\text{contact}}
]

world model auxiliary loss：

[
L_{\text{wm}}
=============

\sum_{i=1}^{N}
|
\psi_s(s_{t+i}) -
\psi_s(\hat{s}_{t+i})
|_1
]

---

# 17. 優先順位

実装順はこれ。

1. **AMP base policyを作る**
2. **baseをfreezeする**
3. **action residual adapterを追加する**
4. **disturbance gateを入れる**
5. **push curriculumでadapterだけをPPO学習する**
6. **world model auxiliary lossを追加する**
7. **failure-aware push miningを入れる**
8. **必要ならlayer-wise adapterに拡張する**
9. **最後にmulti-criticを足す**

multi-criticは最後。
最初にやるべきではない。

この案の本体は、

[
\text{Frozen AMP Base}
+
\text{History-informed Adapter}
+
\text{Recovery-gated AMP}
]

です。
ここを外すと、ただの「AMPにpush rewardを足したpolicy」になって、また同じ失敗をする。

[1]: https://arxiv.org/pdf/2509.13833 "Track Any Motions under Any Disturbances"
