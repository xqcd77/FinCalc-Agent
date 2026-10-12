# 数据与阶段监督

## 来源

原始语料与开发题来自FinanceComplexQA，训练集包含FinAR-Bench改造样本与针对真实失败类型构造的合成／纠错样本。
合成与改造流程使用DeepSeek生成候选，随后执行格式、公式、数值和引用检查；程序审查不等于全部知识经过独立人工验证。
发布包不再分发完整第三方数据或训练集。数据许可请在上游页面确认。

## 一题一条

外层可使用`messages`（system/user/assistant）；训练前转换为`prompt`与`completion`。
同一题的规划、绑定和受限修复保留在一条完整监督轨迹，显式声明当前阶段与对应输出。
不是把一题拆成多条而后混记样本数。

| 阶段 | 输入 | 输出契约 |
|---|---|---|
| plan | 问题、计算需求与格式要求 | formula_id、answer_unit、fields、custom时expression及结构化步骤 |
| bind | 当前动态ID、证据编号、原表与单位 | 当前输入的value/unit/source_ref/quote，必要时unit_quote |
| repair | 原绑定与具体错误、可用证据 | 仅修对应错误；已通过输入不重写；不能返回plan对象替代bind |

原始叶子值必须能在证据中定位。派生指标由表达式计算，不能伪装成来源值。
年份、公司、期间及单位必须与需求对应，百分数只归一化一次，缺失值不填0。

## 训练文件

目录至少包含`train.jsonl`、`validation.jsonl`及两个同名`.metadata.jsonl`、`manifest.json`、`verified.json`。
manifest声明版本、总数、split计数和outputs_sha256；verified声明status=passed与通过计数。
metadata包含唯一question_id、split、training_eligible、来源、问题指纹与分组，按实际数据维护。
入口`load_and_validate_dataset`是当前校验定义，不能只改passed标签绕过检查。

## 数据使用边界

当前163题用于多轮开发，并有真实题背景参与训练补充。因此结果为开发回归表现，不是独立测试集泛化估计。
后续评估泛化应另设独立公司／文档／题目划分，不能与当前数字混用。
