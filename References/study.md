# 学习记录：催化剂嫁接型 LCST 凝胶球的反应热自振荡模型

> 源文件：`gel_sphere_model.md`
> 开始日期：2026-04-05

---

## 一、模型核心思想

### 1.1 物理构型

- **对象**：半径 $R_0$ 的 LCST 热敏凝胶球（如 PNIPAM），浸于恒温浴 $T_\infty$ 中
- **催化剂**：嫁接在聚合物网络上（如 Pt 纳米颗粒），随聚合物运动，不被消耗
- **底物**：外浴中浓度恒定为 $u_b$（如 H₂O₂），扩散进入凝胶后发生放热反应

### 1.2 自振荡闭环机制

```
底物扩散进入 → 催化放热 → T↑ → χ↑ → J↓ (塌缩)
  → D, Mₛ, a(J)↓ → 底物供给受阻 → R↓ → T↓ → J↑ (再膨胀) → 循环
```

**核心亮点**：振荡完全由物理反馈（热-溶胀耦合）驱动，不需要化学振荡器（如 BZ 反应）。

### 1.3 与上一个模型的简化

催化剂固定在聚合物上 → 不需要追踪催化剂浓度 → 主变量减少为三个：
- $J$（溶胀比）
- $u$（底物浓度）
- $T$（温度）

---

## 二、数学框架

### 2.1 球坐标拉格朗日运动学

| 量 | 表达式 | 含义 |
|---|---|---|
| 径向拉伸比 | $\lambda_r = \partial r / \partial R$ | 径向形变 |
| 切向拉伸比 | $\lambda_\theta = r/R$ | 周向形变 |
| 溶胀比 | $J = \lambda_r \cdot \lambda_\theta^2$ | 局部体积比 |
| 聚合物体积分数 | $\phi = \phi_{p0}/J$ | 塌缩时增大 |

**球坐标关键特点**：$\lambda_r \neq \lambda_\theta$（除非 $J$ 空间均匀），会产生剪切应力差。

### 2.2 完整 PDE 方程组（五部分）

**(A) 溶剂守恒（溶胀方程）**
$$\partial_t J = -\frac{1}{R^2}\partial_R(R^2 Q_s), \quad Q_s = -M_s(J,T)\partial_R \mu_s$$

**(B) 底物输运**
$$\partial_t(Ju) + \frac{1}{R^2}\partial_R(R^2 N_u) = -J\mathcal{R}(u,T,J)$$
- 包含对流（溶剂携带）+ 扩散 + 反应消耗

**(C) 热方程**
$$C_{\mathrm{eff}}(J)\partial_t T = \frac{1}{R^2}\partial_R(R^2 K_T(J)\partial_R T) + (-\Delta H_r)J\mathcal{R}$$
- 省略了溶剂携热项（热传导远快于溶剂流动）

**(D) 力学平衡（球坐标特有）**
$$\frac{\partial s_r}{\partial R} + \frac{2}{R}(s_r - \frac{R}{r}s_\theta) = 0$$
- 确定给定 $J(R,t)$ 下的位移场 $r(R,t)$

**(E) 化学势**
$$\mu_s = \mu_{\mathrm{mix}}(J,T) + \mu_{\mathrm{el}}(\lambda_r,\lambda_\theta) - \kappa_J \frac{1}{R^2}\partial_R(R^2 \partial_R J)$$
- Flory-Huggins 混合 + 弹性 + 梯度项

### 2.3 催化反应速率构建

$$\mathcal{R} = \underbrace{k_0 u^n \exp(-E_a/R_g T)}_{\text{本征动力学}} \times \underbrace{\phi_{p0}/J}_{\text{催化剂密度}} \times \underbrace{a(J)}_{\text{可达性因子}}$$

**两个竞争效应**：
- 塌缩(J↓) → 催化剂密度增大（促进反应）
- 塌缩(J↓) → 孔隙减小，底物可达性降低（抑制反应）

可达性因子 $a(J) = (1 - \phi_{p0}/J)^m$，$m \geq 2$ 时净效应是塌缩抑制反应 → 自振荡的必要条件。

### 2.4 边界条件

| 位置 | 条件 |
|---|---|
| 球心 $R=0$ | 对称性：$Q_s=0$, $N_u=0$, $\partial_R T=0$, $\lambda_r=\lambda_\theta$ |
| 外表面 $R=R_0$ | Robin 型：溶剂/底物/热交换 + 自由表面 $s_r=0$ |

---

## 三、集总（0D）模型

### 3.1 前提假设

小尺寸凝胶球，球内梯度可忽略 → 所有变量空间均匀 → 各向同性溶胀 $\lambda_r = \lambda_\theta = J^{1/3}$

### 3.2 三变量 ODE 系统

| 方程 | 物理含义 |
|---|---|
| $dJ/dt \propto J^{2/3}[\mu_{\mathrm{bath}} - \mu_s(J,T)]$ | 溶胀/塌缩动力学 |
| $d(Ju)/dt = \text{表面输入} - \text{反应消耗}$ | 底物进出平衡 |
| $C_{\mathrm{eff}} dT/dt = \text{反应产热} - \text{表面散热}$ | 热平衡 |

### 3.3 关键无量纲参数

| 参数 | 物理意义 |
|---|---|
| $\mathrm{Da}$ (Damköhler数) | 反应速率 vs 溶胀速率 |
| $\mathrm{Bi}_u$ (底物 Biot数) | 底物表面传质 vs 球内扩散 |
| $\mathrm{Bi}_T$ (热 Biot数) | 表面散热 vs 球内导热 |
| $\Gamma_A$ | 温度对反应速率的敏感性 |

---

## 四、自振荡条件分析

### 4.1 反馈环结构

**快正反馈环**（时间尺度 $\tau_T$）：
$$u \xrightarrow{+} \mathcal{R} \xrightarrow{+} T \xrightarrow{+} \chi \xrightarrow{-} J$$

**慢负反馈环**（时间尺度 $\tau_s \gg \tau_T$）：
$$J\!\downarrow \xrightarrow{-} D_u, a(J) \xrightarrow{-} u_{\mathrm{eff}} \xrightarrow{-} \mathcal{R} \xrightarrow{-} T \xrightarrow{+} J\!\uparrow$$

### 4.2 六个必要条件

1. 热源足够强（$\mathrm{Da}$ 足够大）
2. 散热不能太快（$\mathrm{Bi}_T$ 不能太大）
3. 工作点在 LCST 附近
4. 时间尺度分离：$\tau_s/\tau_T \gg 1$
5. 塌缩对反应的抑制足够强
6. 但不能完全锁死（仍需缓慢渗入）

### 4.3 Hopf 分岔

稳态处 Jacobian 矩阵 $\mathbf{A}$ 出现纯虚特征值 → 自振荡出现。

---

## 五、球 vs 平板对比

| 特征 | 平板 | 球 |
|---|---|---|
| 面体比 | $1/H_0$ | $3/R_0$（更大） |
| 力学 | $\lambda_r=J$，无剪切 | $\lambda_r \neq \lambda_\theta$，有剪切 |
| 散热 | 面散热 | 球散热更快 |
| 底物供给 | 单面 | 全表面（更均匀） |
| 塌缩形态 | 表皮层 | 可能 core-shell |

球的矛盾：面体比大 → 供给+散热高效，但也意味着维持温升更难。**球越小越难振荡**。

---

## 六、物理可实现性

### 6.1 候选体系

| 体系 | 反应 | $\Delta H$ | 特点 |
|---|---|---|---|
| PNIPAM + Pt + H₂O₂ | $2H_2O_2 → 2H_2O + O_2$ | −98 kJ/mol | 最有前景，但有气泡问题 |
| PNIPAM + GOx + 葡萄糖 | 葡萄糖→葡萄糖酸+H₂O₂ | −80 kJ/mol | 酶高温失活可增强负反馈 |
| PNIPAM + 过氧化氢酶 + H₂O₂ | 同上 | — | 反应极快，可能集中在表层 |

### 6.2 最优尺寸窗口

$$R_0 \sim 50\text{–}500\;\mu\mathrm{m}$$

- 太大(>1 mm)：溶胀太慢（~天量级）
- 太小(<10 μm)：热散失太快，无法升温

### 6.3 与 Yoshida BZ 凝胶对比

| | BZ 凝胶 | 本模型 |
|---|---|---|
| 振荡源 | BZ 化学振荡 | 热-溶胀耦合 |
| 凝胶角色 | 被动响应 | 主动参与反馈 |
| 是否需要化学振荡器 | 是 | **否** |

---

## 七、可行性分析要点（2026-04-05）

> 详见 `feasibility_analysis.md` 和 `references_compiled.md`

### 7.1 建模可行性总结

| 模块 | 评估 | 关键点 |
|---|---|---|
| Flory-Huggins + Flory-Rehner | ✅ 成熟 | $\chi(T,\phi)$ 拟合是关键 |
| 球坐标力学 | ✅ 可行 | Hong-Suo/Chester-Anand 框架可用 |
| 催化反应速率 | ⚠️ 定性对 | $m$ 指数需实验标定 |
| 溶剂渗流 | ✅ 经典 | $D_{coop} \sim 10^{-11}$ m²/s 已验证 |
| 热方程 | ⚠️ 注意 | $\tau_T \ll \tau_s$，极端时间尺度分离 |
| 0D 集总模型 | ✅ 推荐 | 分岔分析的理想起点 |

### 7.2 实验可行性总结

- **最优尺寸窗口**：$R_0 \sim 50-500$ μm（溶胀~分钟级，热扩散~微秒级）
- **首选体系**：PNIPAM + Pt + H₂O₂（但有 O₂ 气泡问题）
- **替代体系**：PNIPAM + GOx + 葡萄糖（更安全，但酶稳定性待验证）
- **关键先例**：He et al. (2012) Nature — 合成恒温材料实现了化学-力学-热耦合反馈

### 7.3 三大核心不确定性

1. **振荡参数窗口是否足够宽** → 需 0D 模型分岔分析量化
2. **O₂ 气泡问题** → 需选择不产气体系或控制浓度
3. **2-3°C 温升的 Arrhenius 效应是否足够** → 需定量估计

---

## 八、讨论记录

### Q&A 记录

*待补充...*

---

## 九、个人理解与疑问

*待补充...*
