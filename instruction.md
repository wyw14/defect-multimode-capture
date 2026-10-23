# 多模缺陷跃迁谱核

在 `/app/src/vibronic_kernel` 实现 `capture_kernel`。这是基于 pymatgen-analysis-defects 2026.10.16 的作者研究扩展，不是声称上游有已知缺陷或已有实验事故。原生项目提供 HarmonicDefect、配置坐标与单坐标复合计算；这里要求完整内部振动空间的量子跃迁谱核。可使用随包安装的依赖并新增包内源码，不限制合理求解方法。

## 原生几何与物理模型

初、末态均为 pymatgen.core.Structure，晶格相同，元素有序且无占位混合。每个原子有唯一字符串 site property `atom_id`；两态的 ID 集合与对应元素相同，原子次序可以不同。两态原子按 ID 对应。位移取原生晶格度量下从初态到末态的唯一最短周期像；这条周期分支在所请求扰动附近保持不变。整体平移属于自由声学坐标。

初态原子顺序定义 Cartesian 向量 x=(x_1x,x_1y,x_1z,...,x_Nz)，单位 Å，原点为初态平衡结构。对齐后的末态最短周期位移记为 d。两套实对称力常数 Ki、Kf 的单位为 eV/Å²；Ki 按初态原子顺序，Kf 按末态原子顺序，均使用原生结构的全局 Cartesian 轴。它们在对齐后恰有三个整体平移零模，并在内部空间正定。三列 t_alpha 是所有原子同时沿 Cartesian alpha 方向平移的向量，K t_alpha=0。

质量 m 的单位为 amu，缺省使用初态 site.specie.atomic_mass，也可提供按初态顺序的同位素质量数组。M 是将每个原子质量重复三次形成的对角阵。定义 kappa=HBAR_EV**2*EV2J/(AMU2KG*ANGS2M**2)，常数从 pymatgen.analysis.defects.constants 取得；HBAR_EV 表示 eV·s 形式的约化普朗克常数。这与原生 HarmonicDefect.omega_eV 的频率换算口径一致。

在移除三个自由整体平移后的内部核空间，初、末态哈密顿量分别为

Hi=-(kappa/2) nabla_x^T M^-1 nabla_x +(1/2)x^T Ki x；
Hf=-(kappa/2) nabla_x^T M^-1 nabla_x +(1/2)(x-d)^T Kf(x-d)。

上式 Kf 与 d 已按初态 ID 对齐。可任选等价的内部坐标表示，不能把被移除的自由声学模式计入配分函数。初、末态一般不能用同一套独立正常模式同时对角化。振动能量包含零点能。不同内部基、简并模中的基选择、整体 Cartesian 旋转或相同原子结构的周期像表示，不改变物理结果。

电子跃迁耦合为 Hermitian 的实坐标乘法算符 W(x)=w0+g^T x+(1/2)x^T B x。w0 单位 eV，g 单位 eV/Å，B 为实对称阵，单位 eV/Å²，均按初态 Cartesian 顺序。g 与 B 对整体平移不变，即 t_alpha^T g=0、B t_alpha=0。它们与 x 一起相对初态平衡结构定义。

## 要求的物理量

设 Hi 的内部量子本征态为 |a,i>、能量 epsilon_ai，Hf 的内部本征态为 |b,f>、能量 epsilon_bf。温度 T 用 K，beta=1/(KB*T)，KB 从原生 constants 取得，单位 eV/K。Zi=sum_a exp(-beta*epsilon_ai)，Zf 同理。

对每个给定可用电子能量 E，计算

Phi(E)=sum_(a,b) [exp(-beta*epsilon_ai)/Zi] * |<b,f|W|a,i>|² * exp(-(E-(epsilon_bf-epsilon_ai))²/(2*sigma²))/(sqrt(2*pi)*sigma)。

这是对全部内部量子态的求和；sigma 是给定的 Gaussian 能量展宽的标准差，单位 eV。E 可以为负，零点能差包含在振动能量差中。Phi 的单位为 eV，是本题定义的核振动跃迁谱核；没有额外体积、简并度、载流子态密度或 2*pi/hbar 因子，也不将它冒称完整的 cm³/s SRH 捕获系数。

还需返回完整内部模式的初、末态振动能量量子 omega（eV，非负升序，各有 3N-3 个）和 log(Zi)、log(Zf)。内部频率可以简并，不要求输出任何简并模式的特定基。

## API

```python
capture_kernel(initial, final, force_initial, force_final,
               coupling, temperature, energies, width,
               masses=None, queries=())
```

- initial、final 是原生 Structure 对象。
- force_initial、force_final 为 (3N,3N) 数值数组，顺序如上。
- coupling 恰含 constant（w0 标量）、linear（g，3N）、quadratic（B，3N×3N）。
- temperature 为 T，energies 为长度 L 的实数序列，width 为 sigma。
- masses 为 None 或长度 N 的正质量数组，按初态顺序。
- queries 是 q 个方向的 list。每项是 dict，可含 force_initial、force_final、initial_cartesian、final_cartesian、masses、constant、linear、quadratic、temperature、energies、width；缺少字段表示零方向。形状与原输入一致，两个 Cartesian 方向为 (N,3)，各自按所在原生结构顺序，单位 Å。

一个查询是这些原始物理输入的同时一阶变化，不改变晶格、元素、ID 对应关系或选中的最短周期像分支。即使 masses=None，质量方向也表示对缺省原生原子质量的扰动。力常数与耦合方向保持平移不变及矩阵对称性；内部稳定性在局部保持。导数是完整 Phi 对该同时扰动的一阶响应，包含核动能度量、几何、两套势能面、耦合、热归一化、E 和展宽的共同变化。允许精确内部频率简并及会拆分简并的扰动；只要求物理谱核的导数，不要求个别排序简并频率的导数。

返回 dict，恰含 kernel、kernel_response、omega_initial、omega_final、log_Z_initial、log_Z_final。kernel 形状 (L,)；kernel_response 形状 (q,L)，q=0 保留零长度首维；两个 omega 形状 (3N-3,)，两个 log_Z 为实标量。数值输出必须有限，可用 ndarray 或可转换成相同形状的数值序列，不能返回编码数组字符串。

## 支持域、误差与失败行为

2<=N<=6，1<=L<=32，0<=q<=3；200<=T<=1000 K，0.015<=sigma<=0.05 eV，-0.2<=E<=0.35 eV。全部内部振动能量量子位于 [0.025,0.18] eV。质量为 [1,210] amu，晶格可为非正交。结构使用相同固定晶格；最短像唯一且距离对应分支切换至少 10^-5 Å。固定查询的邻域仍有上述物理定义，输入方向有限、保持公开物理约束，连续缩小扰动可保持分支。

在原生质量度量下，末态平衡位移的无量纲重组强度 sum_j omega_fj*dQ_fj²/(2*kappa) 不超过6，dQ_f 是末态质量归一化内部模式上的位移。所有输入为合成稳定谐模型，非实测力常数。热初态的 <W²> 不超过0.1 eV²；请求方向的各输入变化量按相应上述单位计，绝对分量不超过1。绝对平衡位置可以相差周期晶格向量。

kernel 验收 atol=5e-8、rtol=5e-6；kernel_response 验收 atol=1e-6、rtol=2e-5。omega 和 log_Z 验收 atol=1e-7、rtol=3e-6。键集合、类型、形状与有限性单独检查。不得将截断或数值求解失败当作可拒绝的合法输入。

非法字段、形状、非有限数值、非正质量、ID/元素/晶格不匹配、非对称矩阵或违反声学/耦合平移约束时抛出 ValueError；其他支持域和查询物理条件由调用方保证。输入结构、数组、coupling、queries 均不得被修改；拒收时也保持原值。

每次调用限180秒，2 CPU、4 GiB。只有 /app/src 的候选源码转入独立验收环境。候选可使用临时目录，不得依赖参考解、验收脚本、预置答案或修改评分环境获得结果。题目不要求运行 VASP，不依赖材料数据库账户或外部实验设备。
