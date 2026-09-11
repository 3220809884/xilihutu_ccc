# 储能退化成本参考资料

本目录用于支撑“在基础调度模型中加入储能循环退化成本，并与不计退化成本的方案进行对比”这一扩展研究。这里的“退化成本”是电池因循环使用造成的寿命消耗，不等同于充放电效率导致的能量损耗；基础模型已经通过充放电效率处理了后者。

## 1. 已收录全文

### [D1] 线性退化模型与 MILP

CARDOSO G, BROUHARD T, DEFOREST N, et al. Battery aging in multi-energy microgrid design using mixed integer linear programming[J]. *Applied Energy*, 2018, 231: 1059-1069. DOI: [10.1016/j.apenergy.2018.09.185](https://doi.org/10.1016/j.apenergy.2018.09.185).

- 本地全文：[01_Cardoso_2018_Battery_aging_MILP.pdf](01_Cardoso_2018_Battery_aging_MILP.pdf)
- 公开来源：[Lawrence Berkeley National Laboratory](https://ets.lbl.gov/publications/battery-aging-multi-energy-microgrid)
- 可借鉴内容：把电池老化以线性形式纳入 MILP，并比较计及与不计及退化时的储能配置、循环次数和经济性。
- 本题用法：支持将单位能量吞吐退化成本加入问题 1 的目标函数，同时保持原模型为 MILP。

### [D2] 含电池成本的微网能量调度

HOSSAIN M A, POTA H R, SQUARTINI S, et al. Energy scheduling of community microgrid with battery cost using particle swarm optimisation[J]. *Applied Energy*, 2019, 254: 113723. DOI: [10.1016/j.apenergy.2019.113723](https://doi.org/10.1016/j.apenergy.2019.113723).

- 本地全文：[02_Hossain_2019_Community_microgrid_battery_cost.pdf](02_Hossain_2019_Community_microgrid_battery_cost.pdf)
- 公开来源：[Griffith Research Repository](https://research-repository.griffith.edu.au/server/api/core/bitstreams/13edebb9-cf15-4b24-89b1-04ac3b14ac4c/content)
- 可借鉴内容：在社区微网能量调度中显式考虑电池使用成本，并考察运行成本与储能使用之间的权衡。
- 使用边界：该文采用粒子群算法，而本题基础求解器采用 MILP；可引用其“计入电池成本”的思想，不应写成本文采用了粒子群算法。

## 2. 已核实题录，未收入全文

### [D3] 退化与不确定性联合建模

NGUYEN Q M, NGUYEN D L, NGUYEN T K. A mixed-integer linear programming model for microgrid optimal scheduling considering BESS degradation and RES uncertainty[J]. *Journal of Energy Storage*, 2024, 104: 114663. DOI: [10.1016/j.est.2024.114663](https://doi.org/10.1016/j.est.2024.114663).

- 出版方页面：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352152X2404249X)
- 可借鉴内容：将储能退化和可再生能源不确定性纳入微网 MILP，可作为问题 2、3 的后续扩展参考。
- 全文状态：出版方页面可能需要机构订阅，本目录不绕过访问限制保存来源不明的副本。

### [D4] 考虑寿命损耗的储能优化配置

冯紫妍, 许仪勋, 汪凯琳, 等. 考虑寿命损耗的微网电池储能容量优化配置[J]. 电源学报, 2024, 22(1): 101-109. DOI: [10.13234/j.issn.2095-2805.2024.1.101](https://doi.org/10.13234/j.issn.2095-2805.2024.1.101).

- 期刊页面：[电源学报](https://castjournals.cast.org.cn/joweb/dyxb/CN/1154040957029835031)
- 可借鉴内容：放电深度、循环寿命和寿命损耗成本的关系，以及分段线性化后进行优化的思路。
- 全文状态：资源平台可能要求登录，本目录暂只保留已核实的题录和官方链接。

## 3. 对本题的推荐引用方式

- 问题 1 基础模型：仍按题目原始目标最小化计划购电费，不加入题目没有给出的成本。
- 问题 1 扩展分析：引用 [D1]、[D2]，比较“不计循环退化成本”和“计及循环退化成本”。
- 问题 2、3 扩展分析：可引用 [D3]，讨论预测误差、滚动调度和退化成本的共同影响。
- 更精细的寿命模型：引用 [D4]，使用放电深度相关的分段线性成本；这属于进阶方案，不宜在参数不足时直接替代基础模型。

## 4. 引用纪律

1. 最终参考文献只保留正文真正引用的条目。
2. 不能把“充放电效率损失”和“循环寿命退化”写成同一概念。
3. 题目未提供电池更换成本和循环寿命，扩展模型中的参数必须注明数据来源，并进行敏感性分析。
4. 不根据文献替换题目给定的容量、功率、效率、SOC 边界和结算规则。
