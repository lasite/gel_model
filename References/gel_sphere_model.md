# 催化剂嫁接型 LCST 凝胶球的反应热自振荡模型

## 0. 物理构型与核心思想

考虑一个半径为 $R_0$ 的热敏（LCST 型）凝胶球，浸泡在恒温浴（温度 $T_\infty$）中。

**构型要素：**

- 凝胶聚合物网络上嫁接有催化剂离子（如 Pt 纳米颗粒、固定化酶等），催化剂随聚合物一起运动，**不被消耗**
- 外浴中含有底物 $u$（如 $\mathrm{H_2O_2}$、葡萄糖等），浓度恒定为 $u_b$
- 底物从外浴扩散进入凝胶，在催化剂上发生**放热反应**，产物扩散出去
- 凝胶具有 LCST 相变特性（如 PNIPAM），温度升高时 $\chi$ 增大，凝胶塌缩

**核心自振荡闭环：**

$$
u \text{ 扩散进入} \;\xrightarrow{\text{催化}}\; \text{放热} \;\rightarrow\; T\!\uparrow \;\rightarrow\; \chi\!\uparrow \;\rightarrow\; J\!\downarrow\;(\text{塌缩})
$$
$$
\rightarrow\; D,M_s,a(J)\!\downarrow \;\rightarrow\; u\text{ 供给受阻} \;\rightarrow\; R\!\downarrow \;\rightarrow\; T\!\downarrow \;\rightarrow\; J\!\uparrow\;(\text{再膨胀}) \;\rightarrow\; \text{循环}
$$

与上一个模型相比，这里有一个关键的简化：**催化剂是固定的，不需要一个独立的"反应物守恒方程"来追踪催化剂浓度**。系统的主变量减少为三个：$J$（溶胀比）、$u$（底物浓度）、$T$（温度）。

---

## 1. 球坐标拉格朗日运动学

取参考构型的径向物质坐标 $R \in [0, R_0]$。当前径向位置为 $r(R,t)$。

球对称变形的拉伸比为

$$
\lambda_r = \frac{\partial r}{\partial R}, \qquad \lambda_\theta = \frac{r}{R}
$$

局部体积比（溶胀比）为

$$
J(R,t) = \lambda_r \cdot \lambda_\theta^2 = \frac{\partial r}{\partial R} \cdot \left(\frac{r}{R}\right)^2
$$

聚合物体积分数

$$
\phi(R,t) = \frac{\phi_{p0}}{J(R,t)}
$$

当前凝胶球总半径

$$
r_{surface}(t) = r(R_0, t)
$$

**球坐标与平板的关键区别：** 在球中，即使溶胀是纯径向的，径向拉伸 $\lambda_r$ 和切向拉伸 $\lambda_\theta$ 一般不相等（除非 $J$ 空间均匀），这会产生剪切应力差 $\sigma_r - \sigma_\theta \neq 0$，需要额外的力学平衡方程。

---

## 2. 完整的球坐标拉格朗日方程组

### (A) 溶剂守恒 / 溶胀方程

$$
\partial_t J = -\frac{1}{R^2}\,\partial_R\!\left(R^2 Q_s\right)
$$

$$
Q_s = -M_s(J,T)\,\partial_R \mu_s
$$

球坐标的散度 $\frac{1}{R^2}\partial_R(R^2 \cdot)$ 取代了平板中的 $\partial_X$。

### (B) 底物输运方程

$$
\partial_t(J\,u) + \frac{1}{R^2}\,\partial_R\!\left(R^2 N_u\right) = -J\,\mathcal{R}(u,T,J)
$$

$$
N_u = u\,Q_s - D_u(J,T)\,\partial_R u
$$

注意：这里 $u$ 是底物浓度（相对于当前体积），$\mathcal{R}$ 是反应速率。底物被溶剂流对流携带，也在变化的孔隙结构中扩散。

### (C) 热方程

$$
C_{\mathrm{eff}}(J)\,\partial_t T = \frac{1}{R^2}\,\partial_R\!\left(R^2\,K_T(J)\,\partial_R T\right) + (-\Delta H_r)\,J\,\mathcal{R}(u,T,J)
$$

这里省略了溶剂携热项（对于凝胶球，热传导通常远快于溶剂流动）。

### (D) 力学平衡（球坐标特有）

在准静态假设下（力学响应远快于扩散），应力平衡给出

$$
\frac{\partial s_r}{\partial R} + \frac{2}{R}\left(s_r - \frac{R}{r}\,s_\theta\right) = 0
$$

其中 $s_r$、$s_\theta$ 是名义应力（Piola-Kirchhoff）。对于 neo-Hookean 网络加 Flory-Huggins 混合能，它们是 $\lambda_r$、$\lambda_\theta$、$T$ 的已知函数。

这个方程确定了在给定 $J(R,t)$ 分布下的位移场 $r(R,t)$，从而给出 $\lambda_r$ 和 $\lambda_\theta$ 的分配方式。

### (E) 化学势

$$
\mu_s = \mu_{\mathrm{mix}}(J,T) + \mu_{\mathrm{el}}(\lambda_r,\lambda_\theta) - \kappa_J\,\frac{1}{R^2}\,\partial_R\!\left(R^2\,\partial_R J\right)
$$

混合部分仍采用 Flory-Huggins + 温度依赖 $\chi$：

$$
\mu_{\mathrm{mix}} = \frac{R_g T}{v_s}\left[\ln(1-\phi) + \phi + \chi(T,\phi)\,\phi^2\right]
$$

$$
\chi(T,\phi) = \chi_0(T) + \chi_1\,\phi
$$

弹性部分在球坐标中更复杂，因为 $\lambda_r \neq \lambda_\theta$：

$$
\mu_{\mathrm{el}} = \frac{G\,v_s}{J}\left(\frac{\lambda_r^2 + 2\lambda_\theta^2}{3} - 1\right) + \cdots
$$

（具体形式取决于所选网络模型。）

---

## 3. 催化反应速率的合理构建

催化剂嫁接在聚合物上，这意味着：

**催化剂浓度不是独立变量**——它与聚合物密度成正比。

设参考态催化剂浓度为 $c_{\mathrm{cat},0}$（per reference volume，固定不变），则当前态催化剂浓度为 $c_{\mathrm{cat},0}/J$。当凝胶塌缩（$J\downarrow$）时，催化剂密度反而增大。

但同时，塌缩使孔隙变小，底物到达催化位点的可达性降低。

因此反应速率应包含两个竞争效应：

$$
\mathcal{R}(u,T,J) = \underbrace{k_0\,u^n\,\exp\!\left(-\frac{E_a}{R_g T}\right)}_{\text{本征动力学}} \;\times\; \underbrace{\frac{\phi_{p0}}{J}}_{\text{催化剂密度} \propto \phi} \;\times\; \underbrace{a(J)}_{\text{可达性因子}}
$$

可达性因子 $a(J)$ 必须在塌缩时**急剧下降**，以压过催化剂密度增大的效应：

$$
a(J) = \left(1 - \frac{\phi_{p0}}{J}\right)^m, \qquad m \geq 2
$$

净效应：定义有效反应因子

$$
f(J) = \frac{\phi_{p0}}{J} \cdot a(J) = \frac{\phi_{p0}}{J}\left(1 - \frac{\phi_{p0}}{J}\right)^m
$$

当 $m \geq 2$ 时，$f(J)$ 在 $J$ 减小到某个阈值以下后急剧下降，即**塌缩最终抑制反应**。这是自振荡的必要条件。

另外，底物扩散系数也要随塌缩下降：

$$
D_u(J) = D_0\left(1 - \frac{\phi_{p0}}{J}\right)^{m_D}
$$

---

## 4. 边界条件

### 球心 $R=0$（对称性）

$$
Q_s = 0, \qquad N_u = 0, \qquad \partial_R T = 0, \qquad \partial_R r = \frac{r}{R}\;(\text{即}\;\lambda_r=\lambda_\theta)
$$

最后一条确保球心没有奇异性。

### 外表面 $R=R_0$

**溶剂交换：**
$$
Q_s = L_\mu\,\big(\mu_s - \mu_{\mathrm{bath}}\big)
$$

**底物交换：**
$$
N_u = k_u\,(u - u_b)
$$

**换热：**
$$
-K_T\,\partial_R T = h_T\,(T - T_\infty)
$$

**力学边界：** 自由表面，名义应力 $s_r = 0$。

---

## 5. 集总（0D）模型：最简自振荡分析

完整的空间分辨模型计算量大。对于**小尺寸凝胶球**（$R_0$ 足够小，使得球内梯度可以忽略），可以做集总（lumped）近似，将所有变量视为空间均匀的。

此时各向同性溶胀 $\lambda_r = \lambda_\theta = J^{1/3}$，当前半径 $r = R_0\,J^{1/3}$，表面积/参考体积比为

$$
\frac{A_{\mathrm{current}}}{V_{\mathrm{ref}}} = \frac{4\pi R_0^2 J^{2/3}}{(4/3)\pi R_0^3} = \frac{3}{R_0}\,J^{2/3}
$$

### 集总方程组

**(i) 溶胀动力学**

$$
\frac{dJ}{dt} = \frac{3}{R_0}\,J^{2/3}\,L_\mu\,\big[\mu_{\mathrm{bath}} - \mu_s(J,T)\big]
$$

其中 $\mu_s(J,T) = \mu_{\mathrm{mix}}(J,T) + \mu_{\mathrm{el}}(J)$。

当 $\mu_s > \mu_{\mathrm{bath}}$ 时，溶剂倾向于离开凝胶（塌缩）；反之膨胀。

**(ii) 底物动力学**

底物在球内近似均匀，变化来自：表面输入 - 反应消耗 - 体积变化稀释

$$
\frac{d(Ju)}{dt} = \frac{3}{R_0}\,J^{2/3}\,k_u\,(u_b - u) - J\,\mathcal{R}(u,T,J)
$$

展开：

$$
J\frac{du}{dt} = -u\frac{dJ}{dt} + \frac{3}{R_0}\,J^{2/3}\,k_u\,(u_b - u) - J\,\mathcal{R}(u,T,J)
$$

**(iii) 热平衡**

$$
C_{\mathrm{eff}}(J)\,\frac{dT}{dt} = (-\Delta H_r)\,\mathcal{R}(u,T,J) - \frac{3}{R_0}\,J^{2/3}\,h_T\,(T - T_\infty)
$$

这就是一个三变量 ODE 系统 $(J, u, T)$，可以直接做相平面分析和分岔分析。

### 集总模型的无量纲化

令

$$
\tau_s = \frac{R_0}{3\,L_\mu\,\mu_*}, \qquad \mu_* = \frac{R_g T_\infty}{v_s}
$$

$$
\tilde{u} = \frac{u}{u_b}, \qquad \theta = \frac{T - T_\infty}{\Delta T_*}, \qquad \Delta T_* = \frac{(-\Delta H_r)\,u_b}{C_*}
$$

$$
\tilde{t} = \frac{t}{\tau_s}
$$

则

$$
\boxed{\frac{dJ}{d\tilde{t}} = J^{2/3}\left[m_{\mathrm{bath}} - m(J,\theta)\right]}
$$

$$
\boxed{J\frac{d\tilde{u}}{d\tilde{t}} = -\tilde{u}\frac{dJ}{d\tilde{t}} + \mathrm{Bi}_u\,J^{2/3}(1-\tilde{u}) - \mathrm{Da}\,J\,\widehat{\mathcal{R}}(\tilde{u},\theta,J)}
$$

$$
\boxed{\mathcal{C}(J)\frac{d\theta}{d\tilde{t}} = \mathrm{Da}\,\widehat{\mathcal{R}}(\tilde{u},\theta,J) - \mathrm{Bi}_T\,J^{2/3}\,\theta}
$$

其中

$$
\mathrm{Da} = \frac{\tau_s\,R_0}{c_b}\,k_0\,u_b^n\,\frac{\phi_{p0}}{J_*}\,\exp\!\left(-\frac{E_a}{R_g T_\infty}\right)
$$

$$
\mathrm{Bi}_u = \frac{k_u\,\tau_s}{R_0/3}, \qquad \mathrm{Bi}_T = \frac{h_T\,\tau_s}{C_*\,R_0/3}
$$

$$
\widehat{\mathcal{R}} = \tilde{u}^n\,f(J)\,\exp\!\left(\frac{\Gamma_A\,\theta}{1+\varepsilon_T\theta}\right)
$$

---

## 6. 自振荡的条件分析

### 6.1 稳态

稳态要求 $dJ/dt = 0$，$du/dt = 0$，$dT/dt = 0$，即

$$
m(J_{ss}, \theta_{ss}) = m_{\mathrm{bath}}
$$
$$
\mathrm{Bi}_u\,J_{ss}^{2/3}(1 - u_{ss}) = \mathrm{Da}\,J_{ss}\,\widehat{\mathcal{R}}_{ss}
$$
$$
\mathrm{Da}\,\widehat{\mathcal{R}}_{ss} = \mathrm{Bi}_T\,J_{ss}^{2/3}\,\theta_{ss}
$$

第一条是溶胀平衡（化学势等于外浴），第二条是底物进出平衡，第三条是产热=散热。

### 6.2 Hopf 分岔与自振荡窗口

在稳态点做线性化，Jacobian 为

$$
\mathbf{A} = \begin{pmatrix}
\partial \dot{J}/\partial J & \partial \dot{J}/\partial u & \partial \dot{J}/\partial T \\
\partial \dot{u}/\partial J & \partial \dot{u}/\partial u & \partial \dot{u}/\partial T \\
\partial \dot{T}/\partial J & \partial \dot{T}/\partial u & \partial \dot{T}/\partial T
\end{pmatrix}_{ss}
$$

自振荡出现在 Hopf 分岔处，需要 $\mathbf{A}$ 有一对纯虚特征值。

关键的矩阵元素物理意义：

- $\partial \dot{T}/\partial u > 0$：更多底物 → 更多反应 → 更多热 （**正耦合**）
- $\partial \dot{J}/\partial T < 0$：温度升高 → $\chi$ 增大 → 塌缩趋势 （**正反馈环的一部分**）
- $\partial \dot{u}/\partial J > 0$：塌缩 → 底物供应受阻 → $u$ 下降 （**负反馈**）
- $\partial \dot{T}/\partial T < 0$：散热 （**自稳定**）

**快正反馈回路**（需要时间尺度 $\tau_T$）：

$$
u \xrightarrow{+} \mathcal{R} \xrightarrow{+} T \xrightarrow{+} \chi \xrightarrow{-} J
$$

**慢负反馈回路**（需要时间尺度 $\tau_s \gg \tau_T$）：

$$
J\!\downarrow \;\xrightarrow{-}\; D_u, a(J) \;\xrightarrow{-}\; u_{\mathrm{eff}} \;\xrightarrow{-}\; \mathcal{R} \;\xrightarrow{-}\; T \;\xrightarrow{+}\; J\!\uparrow
$$

### 6.3 振荡的必要条件

自振荡窗口要求以下条件同时满足：

1. **热源足够强**：$\mathrm{Da}$ 足够大，使反应热能显著升温
2. **热散失不能太快**：$\mathrm{Bi}_T$ 不能太大，否则温升被立刻抹平
3. **工作点在 LCST 附近**：$\chi_\infty$ 和 $\phi_{p0}$ 使平衡态接近相变区
4. **时间尺度分离**：热扩散（快）$\ll$ 溶胀渗流（慢），即 $\alpha = \tau_s/\tau_T \gg 1$
5. **塌缩对反应的抑制必须足够强**：$f(J)$ 在塌缩态显著下降
6. **但不能完全锁死**：塌缩后仍有缓慢的底物渗入，否则系统进入永久塌缩

用物理语言：**凝胶球需要能在"膨胀-高反应"态和"塌缩-低反应"态之间来回切换，且两个切换过程的时间尺度足够不同。**

---

## 7. 与 slab 模型的关键区别

| 特征 | 1D 平板 | 球 |
|------|---------|------|
| 面体比 | $1/H_0$ | $3/R_0$，随塌缩变化 |
| 力学 | $\lambda_r = J$，无剪切 | $\lambda_r \neq \lambda_\theta$，有法向-切向应力差 |
| 散热效率 | 面散热 | 球散热，面体比更大 → 散热更快 |
| 底物供给 | 单面 | 全表面包围 → 供给更均匀 |
| 塌缩形态 | 表皮层塌缩 | 可能形成 core-shell 结构（硬壳-软核） |

球的一个**有利特征**是：面体比大，表面供给和散热都更高效。但这也意味着要维持足够的温升更难——**球越小越难自振荡**，因为热太容易散掉。

存在一个**最优尺寸窗口**：太大，反应物来不及扩散到球心；太小，热量来不及积累就散掉了。

---

## 8. 物理可实现性分析

### 8.1 候选体系

最有可能实现这个机制的体系：

**(a) PNIPAM + 固定化铂纳米颗粒 + H₂O₂ 底物**

- 反应：$2\mathrm{H_2O_2} \xrightarrow{Pt} 2\mathrm{H_2O} + \mathrm{O_2}$
- 反应焓：$\Delta H \approx -98 \;\mathrm{kJ/mol}$
- PNIPAM 的 LCST $\approx 32°\mathrm{C}$
- 外浴温度设在 $T_\infty \approx 30°\mathrm{C}$，仅需 $\Delta T \sim 2\text{–}3°\mathrm{C}$ 即可触发相变

热估算：若底物 $c_b \sim 0.1\;\mathrm{M}$，$\Delta T_{\mathrm{adiabatic}} \approx \frac{98000 \times 100}{4.2\times10^6} \approx 2.3°\mathrm{C}$

这个量级刚好够用。$c_b \sim 1\;\mathrm{M}$ 则有 $\sim 23°\mathrm{C}$ 的裕量。

**(b) PNIPAM + 固定化葡萄糖氧化酶 + 葡萄糖底物**

- 反应：葡萄糖 $\rightarrow$ 葡萄糖酸 + H₂O₂（放热）
- 反应焓：$\Delta H \approx -80 \;\mathrm{kJ/mol}$
- 酶催化效率高，但酶活性可能受温度影响（高温失活反而有助于形成负反馈）

**(c) PNIPAM + 固定化过氧化氢酶 + H₂O₂**

- 过氧化氢酶是已知最高效的酶之一（$k_{\mathrm{cat}} \sim 10^7 \;\mathrm{s^{-1}}$）
- 反应速率极快，可能导致反应集中在表面薄层

### 8.2 关键可行性挑战

**挑战 1：热散失 vs. 热积累**

凝胶球在水浴中的换热系数 $h_T$ 很大（水的自然对流 $h \sim 100\text{–}500\;\mathrm{W/(m^2\cdot K)}$）。凝胶球越小，面体比越大，热越难积累。

Biot 数估算：

$$
\mathrm{Bi}_T = \frac{h_T R_0}{K_{\mathrm{gel}}} \approx \frac{300 \times R_0}{0.5} = 600\,R_0
$$

对 $R_0 = 1\;\mathrm{mm}$，$\mathrm{Bi}_T \approx 0.6$，这意味着球内温度近似均匀，但表面散热仍然很快。

要维持 $\Delta T \sim 2°\mathrm{C}$，需要产热功率密度

$$
\dot{q} \sim h_T \cdot \frac{3}{R_0} \cdot \Delta T \sim 300 \times 3000 \times 2 = 1.8 \;\mathrm{MW/m^3}
$$

这需要 $R \cdot (-\Delta H_r) \sim 1.8 \;\mathrm{MW/m^3}$，即反应速率 $R \sim 18\;\mathrm{mol/(m^3\!\cdot\!s)}$。

对于 $c_b = 0.1\;\mathrm{M}$ 和合理的催化速率，这是可以实现的，但需要足够高的催化剂负载量。

**挑战 2：溶胀时间尺度**

凝胶溶胀的特征时间

$$
\tau_s \sim \frac{R_0^2}{D_{\mathrm{coop}}} \sim \frac{(10^{-3})^2}{10^{-11}} \sim 10^5 \;\mathrm{s}
$$

对于毫米级凝胶球，溶胀需要 $\sim$ 天的量级，太慢了。

对于微米级球（$R_0 \sim 100\;\mu\mathrm{m}$）：
$$
\tau_s \sim \frac{(10^{-4})^2}{10^{-11}} \sim 10^3 \;\mathrm{s} \sim 15\;\text{min}
$$

这个时间尺度更合理。

**结论：自振荡的最优尺寸窗口大约在 $R_0 \sim 50\text{–}500\;\mu\mathrm{m}$。**

- 太大（$>1\;\mathrm{mm}$）：溶胀太慢，振荡周期不切实际
- 太小（$<10\;\mu\mathrm{m}$）：热散失太快，无法维持足够温升

**挑战 3：氧气气泡（针对 H₂O₂ 分解）**

H₂O₂ 分解产生 O₂ 气泡，可能：
- 在凝胶内形成气泡，破坏结构
- 改变局部力学和传质
- 为振荡引入额外的非线性

如果选择不产气的反应（如某些酶催化的水解反应），可以避免这个问题。

### 8.3 与已知自振荡凝胶的对比

目前实验上最成功的自振荡凝胶是 **吉田（Yoshida）的 BZ 反应凝胶**：

- 将 BZ 反应的催化剂 Ru(bpy)₃ 嫁接在 PNIPAM 上
- 自振荡机制是 **BZ 化学振荡器** 本身提供的，不是热反馈
- 凝胶的体积变化是被化学振荡器 *驱动* 的，凝胶本身不参与振荡反馈

**本模型提出的机制与 Yoshida 凝胶的根本区别：**

| | BZ 凝胶（Yoshida） | 本模型（催化热反馈） |
|---|---|---|
| 振荡器 | BZ 化学振荡 | 热-溶胀耦合 |
| 凝胶角色 | 被动响应者 | 主动参与反馈环 |
| 反馈变量 | 化学浓度 | 温度 |
| 需要化学振荡器？ | 是 | 否 |
| 振荡周期 | $\sim$ 分钟（化学钟控制） | $\sim$ 分钟到小时（溶胀+热传输控制） |

本模型的**最大优势**在于：不需要复杂的化学振荡器（如 BZ 反应），只需要一个简单的放热催化反应。振荡完全由**物理反馈**（热-溶胀耦合）驱动。

### 8.4 综合判断

**这个机制在物理上是自洽的，在原理上是可行的。**

主要的实验障碍不是原理层面的，而是工程层面的参数窗口可能较窄：

1. 外浴温度必须精确控制在 LCST 附近（$\pm 1°\mathrm{C}$）
2. 催化剂负载量、底物浓度、凝胶尺寸必须落在合适的窗口
3. 散热条件（搅拌、浴体积）需要匹配

但这些都是可以通过实验优化来解决的。

---

## 9. 推荐的下一步

### 理论方面

1. **先分析 0D 集总模型**：画出 $(Da, S_\chi)$ 平面上的 Hopf 分岔曲线
2. **确定振荡窗口**：在哪些参数组合下存在稳定极限环
3. **估算振荡周期和振幅**

### 数值方面

4. **实现 1D 球坐标的空间分辨模型**：用有限差分或有限元
5. **观察 core-shell 塌缩动力学**：是否形成行进的塌缩前沿

### 实验方面

6. **选择 PNIPAM + 金纳米颗粒 + H₂O₂ 体系**
7. **制备 $\sim 200\;\mu\mathrm{m}$ 的微凝胶球**
8. **在 $T_\infty = 30°\mathrm{C}$ 的 H₂O₂ 溶液中观察体积变化**
