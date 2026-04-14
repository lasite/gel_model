# 参考文献汇编：催化凝胶球自振荡模型

> 整理日期：2026-04-05
> 按主题分类，附文献来源与关键信息

---

## A. 凝胶热力学与相变理论

### [R1] Flory-Rehner 理论对 PNIPAM 微凝胶溶胀的适用性
- **标题**: Does Flory–Rehner theory quantitatively describe the swelling of thermoresponsive microgels?
- **作者**: Nigro et al.
- **期刊**: Soft Matter, 2017
- **DOI/链接**: https://pubs.rsc.org/en/content/articlelanding/2017/sm/c7sm01274h
- **关键内容**: 评估了 Flory-Rehner 理论对 PNIPAM 微凝胶体积相变的定量描述能力，指出标准理论在相变区域的局限性。

### [R2] 改进的 Flory-Rehner 理论用于共聚物微凝胶热致溶胀转变
- **标题**: Modified Flory–Rehner Theory Describes Thermotropic Swelling Transition of Copolymer Microgels
- **作者**: Various
- **期刊**: Polymers (MDPI), 2022, 14(10), 1999
- **DOI/链接**: https://www.mdpi.com/2073-4360/14/10/1999
- **关键内容**: 引入 Hill-like 协作模型改进 $\chi(T,\phi)$ 描述，更好拟合 PNIPAM 共聚物微凝胶的尖锐相变。

### [R2b] Revisiting the Flory–Rehner equation
- **标题**: Revisiting the Flory–Rehner equation: taking a closer look at the Flory–Huggins interaction parameter
- **期刊**: Polymer Bulletin, 2021
- **DOI/链接**: https://link.springer.com/article/10.1007/s00289-021-03836-1
- **关键内容**: 深入分析 $\chi$ 参数对溶胀预测的敏感性。

---

## B. 凝胶力学——扩散-变形耦合理论

### [R3] Hong-Zhao-Suo 凝胶扩散-大变形耦合理论
- **标题**: A Theory of Coupled Diffusion and Large Deformation in Polymeric Gels
- **作者**: Hong W, Zhao X, Zhou J, Suo Z
- **期刊**: Journal of the Mechanics and Physics of Solids, 2008, 56(5): 1779–1793
- **关键内容**: 建立了凝胶中溶剂扩散与大变形耦合的连续介质框架，是当前凝胶力学建模的基础性文献。化学势驱动的溶剂迁移与 neo-Hookean 弹性耦合。

### [R4] Chester-Anand 弹性体流体渗透与大变形耦合理论
- **标题**: A coupled theory of fluid permeation and large deformations for elastomeric materials
- **作者**: Chester SA, Anand L
- **期刊**: Journal of the Mechanics and Physics of Solids, 2010, 58(11): 1879–1906
- **关键内容**: 将 Hong-Suo 框架扩展到更一般的本构关系，提供了有限元实现的完整框架，适合数值求解。

### [R4b] 凝胶相分离的力学基础
- **标题**: Mechanics Underpinning Phase Separation of Hydrogels
- **期刊**: Macromolecules (ACS), 2023
- **DOI/链接**: https://pubs.acs.org/doi/10.1021/acs.macromol.2c02356
- **关键内容**: 讨论力学约束对凝胶相分离的影响，与 core-shell 塌缩结构直接相关。

---

## C. 凝胶溶胀动力学

### [R5] PNIPAM 凝胶球的体积相变动力学与 core-shell 结构
- **标题**: Volume phase transition kinetics of smart N-n-propylacrylamide microgel particles
- **期刊**: Scientific Reports (Nature), 2018
- **DOI/链接**: https://www.nature.com/articles/s41598-018-31976-4
- **关键内容**: 实验观测了微凝胶球的塌缩动力学，证实了"表皮层"形成对塌缩速率的抑制效应。塌缩比膨胀慢得多。

### [R5b] 纳凝胶实时溶胀-塌缩动力学
- **标题**: Real-time swelling-collapse kinetics of nanogels driven by LCST transition
- **期刊**: Science Advances (AAAS), 2024
- **DOI/链接**: https://www.science.org/doi/10.1126/sciadv.adm7876
- **关键内容**: 高时间分辨率测量纳凝胶的溶胀-塌缩动力学。

### [R8] PNIPAM 微凝胶的协作扩散系数
- **标题**: Structure and polymer dynamics within PNIPAM-based microgel particles
- **期刊**: Advances in Colloid and Interface Science, 2014
- **DOI/链接**: https://www.sciencedirect.com/science/article/pii/S0001868613001462
- **关键内容**: 报道了 $D_{coop} \sim 1-4 \times 10^{-11}$ m²/s，与模型中使用的量级一致。DLS 和中子散射测量结果吻合。

### [R8b] Tanaka-Fillmore 凝胶溶胀动力学原始论文
- **标题**: Kinetics of swelling of gels
- **作者**: Tanaka T, Fillmore DJ
- **期刊**: Journal of Chemical Physics, 1979, 70(3): 1214
- **DOI/链接**: https://pubs.aip.org/aip/jcp/article/70/3/1214/89224
- **关键内容**: 建立了凝胶溶胀的孔弹性模型，$\tau \sim R^2/D_{coop}$，是溶胀动力学的基础文献。

### [R8c] 非线性 vs 线性溶胀动力学理论比较
- **标题**: Swelling kinetics of polymer gels: Comparison of linear and nonlinear theories
- **作者**: Huang R et al.
- **期刊**: Soft Matter, 2012
- **DOI/链接**: https://www.researchgate.net/publication/255765713
- **关键内容**: 系统比较了 Tanaka-Fillmore 线性理论与非线性扩展，讨论了大变形下的适用范围。

---

## D. 自振荡凝胶

### [R11] He et al. 合成恒温材料——化学-力学-化学自调节
- **标题**: Synthetic homeostatic materials with chemo-mechano-chemical self-regulation
- **作者**: He X, Aizenberg M, Kuksenok O, Zarzar LD, Shastri A, Musber AC, Olber T, Aizenberg J
- **期刊**: Nature, 2012, 487: 214-218
- **DOI**: 10.1038/nature11223
- **DOI/链接**: https://www.nature.com/articles/nature11223
- **关键内容**: ⭐**与本模型最直接相关的实验文献**。设计了热响应凝胶+催化剂微结构系统，实现了自调节温度恒定。证明了化学-力学-热耦合反馈的实验可行性。本模型可视为 He 系统的"内化"版本（催化剂嵌入凝胶内部而非外部）。

### [R12] Yoshida 自振荡凝胶综述——BZ 反应驱动
- **标题**: Evolution of self-oscillating polymer gels as autonomous polymer systems
- **作者**: Yoshida R, Ueki T
- **期刊**: NPG Asia Materials, 2014, 6: e107
- **DOI/链接**: https://www.nature.com/articles/am201432
- **关键内容**: 全面综述 BZ 驱动的自振荡凝胶发展历程。凝胶是被动响应者，振荡由 BZ 化学钟驱动。

### [R13] Yoshida 自振荡凝胶原始论文
- **标题**: Self-Oscillating Gel
- **作者**: Yoshida R, Takahashi T, Yamaguchi T, Ichijo H
- **期刊**: Journal of the American Chemical Society, 1996, 118: 5134-5135
- **关键内容**: 首次报道 BZ 反应驱动的自振荡凝胶。

### [R14] Zhang et al. 反馈控制凝胶——恒温振荡与耗散信号转导
- **标题**: Feedback-controlled hydrogels with homeostatic oscillations and dissipative signal transduction
- **期刊**: Nature Nanotechnology, 2022
- **DOI/链接**: https://www.nature.com/articles/s41565-022-01241-x
- **关键内容**: 展示了反馈控制凝胶系统的恒温振荡能力，为非 BZ 自振荡凝胶提供了实验先例。

### [R15] Borckmans et al. 自振荡微型凝胶模型
- **标题**: A Model for Self-Oscillating Miniaturized Gels
- **作者**: Borckmans P et al.
- **DOI/链接**: https://nlpc.ulb.be/pdf/03.borckmans.gel.pdf
- **关键内容**: 建立了凝胶振荡器的数学模型，包含 Hopf 分岔和热-力学耦合分析，与本模型的理论框架高度相关。

---

## E. PNIPAM + 催化纳米颗粒复合材料

### [R6] PNIPAM 水凝胶与金属纳米颗粒——可切换催化活性
- **标题**: Kinetic and Equilibrium Function and Switchable Catalytic Activity of Some Thermo-Responsive Hydrogel Metal Absorbents Based on Modified PNIPAM
- **期刊**: ResearchGate, 2022
- **DOI/链接**: https://www.researchgate.net/publication/365296866
- **关键内容**: 实验证明 PNIPAM 的温度响应可以调控嵌入金属纳米颗粒的催化活性——加热使凝胶塌缩，催化位点被屏蔽。直接支持模型中 $a(J)$ 因子的物理图像。

### [R9] PNIPAM + 双金属纳米颗粒复合水凝胶
- **标题**: Synthesis of bimetallic nanoparticles loaded on to PNIPAM hybrid microgel and their catalytic activity
- **期刊**: Scientific Reports (Nature), 2021
- **DOI/链接**: https://www.nature.com/articles/s41598-021-94177-6
- **关键内容**: 报道了在 PNIPAM 微凝胶中原位合成金属纳米颗粒的方法。

### [R9b] PNIPAM 水凝胶综述——从设计到应用
- **标题**: Recent Advances in Poly(N-isopropylacrylamide) Hydrogels and Derivatives
- **期刊**: Advanced Engineering Materials, 2023
- **DOI/链接**: https://advanced.onlinelibrary.wiley.com/doi/10.1002/adem.202201303
- **关键内容**: 全面综述 PNIPAM 水凝胶的最新进展，包括催化、生物传感等应用。

---

## F. 气泡问题与相关挑战

### [R10] 催化 H₂O₂ 分解的 O₂ 气泡生长与脱离
- **标题**: Growth and Detachment of Oxygen Bubbles Induced by Gold-Catalyzed Decomposition of Hydrogen Peroxide
- **期刊**: Journal of Physical Chemistry C (ACS), 2017
- **DOI/链接**: https://pubs.acs.org/doi/10.1021/acs.jpcc.7b04994
- **关键内容**: 详细研究了催化 H₂O₂ 分解中气泡的成核、生长和脱离动力学。对评估气泡对凝胶内催化反应的影响至关重要。

### [R10b] 利用 H₂O₂ 在水凝胶中引入大孔结构
- **标题**: Inducing Macroporosity in Hydrogels using Hydrogen Peroxide as a Blowing Agent
- **期刊**: Materials Chemistry Frontiers (RSC), 2017
- **DOI/链接**: https://pubs.rsc.org/en/content/getauthorversionpdf/C6QM00052E
- **关键内容**: 利用 H₂O₂ 分解产气在凝胶中制造大孔——反面说明了气泡对凝胶结构的破坏性。

---

## G. PNIPAM 基础物性参数

### [R7] 热响应水凝胶综述——PNIPAM 物性
- **标题**: Smart Poly(N-isopropylacrylamide)-Based Hydrogels: A Tour D'Horizon of Biomedical Applications
- **期刊**: Gels (MDPI), 2025, 11(3): 207
- **DOI/链接**: https://www.mdpi.com/2310-2861/11/3/207
- **关键内容**: 全面综述 PNIPAM 基础物性（LCST、溶胀比、力学性能等），可作为模型参数取值的参考。

### [R7b] PNIPAM 智能水凝胶——设计、性能与应用
- **标题**: Poly(N-isopropylacrylamide)-based smart hydrogels: Design, properties and applications
- **期刊**: Progress in Materials Science, 2020
- **DOI/链接**: https://www.sciencedirect.com/science/article/pii/S0079670020300669
- **关键内容**: 涵盖 PNIPAM 水凝胶的设计原则和关键性能参数。

---

## 文献使用建议

| 用途 | 首选文献 |
|---|---|
| $\chi(T)$ 参数取值 | R1, R2, R7b |
| 凝胶力学建模框架 | R3, R4 |
| 溶胀时间尺度估计 | R8, R8b |
| 自振荡实验先例 | R11 (He), R12 (Yoshida), R14 (Zhang) |
| 催化活性可切换性 | R6 |
| Core-shell 塌缩实验 | R5, R5b |
| 0D 模型分岔分析方法 | R15 (Borckmans) |
| 气泡问题评估 | R10, R10b |
